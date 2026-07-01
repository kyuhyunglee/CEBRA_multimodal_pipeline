import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipConnectionBlock(nn.Module):
    """
    CEBRA 뼈대에서 사용되는 커널 사이즈 3의 1D Conv + Skip Connection 블록
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # 시간 차원(Length)을 유지하기 위해 padding=1 적용
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: (Batch, Hidden, Time)
        residual = x
        x = self.conv(x)
        # 논문 구조에 따라 GELU 적용 후 skip connection 더하기
        x = F.gelu(x) 
        return x + residual


class ProbeEncoder(nn.Module):
    """
    고주파 프로브(Neuropixels) 데이터를 위한 인코더
    - 입력 Shape: (Batch, input_neurons, time_window=40)
    """
    def __init__(self, input_neurons, hidden_dim=64, latent_dim=16):
        super().__init__()
        
        # ==========================================
        # 1. 쥐마다 새로 학습해야 하는 첫 번째 파트 (Learnable Downsampling)
        # ==========================================
        # 논문 명시: kernel size 4, stride 2 -> Time: 40 -> 20
        self.downsample1 = nn.Conv1d(input_neurons, hidden_dim, kernel_size=4, stride=2, padding=1)
        
        # 논문 명시: activation 없이 바로 이어서 kernel size 3, stride 2 -> Time: 20 -> 10
        self.downsample2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)


        # ==========================================
        # 2. Pre-trained 가중치를 덮어씌울 공통 뼈대
        # ==========================================
        # 3개의 스킵 커넥션 블록
        self.shared_block1 = SkipConnectionBlock(hidden_dim)
        self.shared_block2 = SkipConnectionBlock(hidden_dim)
        self.shared_block3 = SkipConnectionBlock(hidden_dim)
        
        # 최종 투영 (커널 사이즈 3)
        self.shared_final = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)


    def forward(self, x):
        # 파이토치 Conv1d를 위해 (Batch, Time, Neurons) -> (Batch, Neurons, Time)으로 변경
        x = x.transpose(1, 2)
        
        # 1. Learnable Downsampling (활성화 함수 없이 연속 적용 후 GELU)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = F.gelu(x)
        
        # 2. Shared Backbone (ResNet 구조)
        x = self.shared_block1(x)
        x = self.shared_block2(x)
        x = self.shared_block3(x)
        
        # 3. Final Projection
        x = self.shared_final(x)
        
        # 시계열 차원(Time)을 평균내어 최종 (Batch, latent_dim) 벡터로 압축
        x = torch.mean(x, dim=2)
        
        # CEBRA 특징: L2 Normalization (초구면 매핑)
        return F.normalize(x, p=2, dim=1)


    def load_pretrained_backbone(self, weight_path, freeze=False):
        """
        Pre-trained 모델에서 뉴런 수에 의존하는 downsample 층을 제외하고,
        shared_block 및 shared_final 가중치만 선별적으로 가져옵니다.
        """
        pretrained_dict = torch.load(weight_path)
        model_dict = self.state_dict()
        
        # 'shared_'라는 이름이 들어간 파라미터만 필터링하여 이식
        transfer_dict = {k: v for k, v in pretrained_dict.items() 
                         if 'shared_' in k and k in model_dict}
        
        model_dict.update(transfer_dict)
        self.load_state_dict(model_dict)
        print(f" 성공: {len(transfer_dict)}개의 공통 뼈대 가중치를 로드했습니다.")
        
        if freeze:
            for name, param in self.named_parameters():
                if 'shared_' in name:
                    param.requires_grad = False
            print(" 공통 뼈대 동결 완료. Downsample 레이어만 학습합니다.")