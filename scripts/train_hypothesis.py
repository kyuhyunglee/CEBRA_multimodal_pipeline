import cebra
import numpy as np
import yaml
import os
import torch
from models.encoders import ProbeEncoder # 또는 CalciumEncoder 사용

def run_hypothesis_training():
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 연구실 데이터 특성에 맞게 인코더 초기화
    # 이제 input_neurons는 실제 측정된 뉴런 개수로 config에 정의되어 있어야 함
    encoder = ProbeEncoder(
        input_neurons=config['dataset']['input_neurons'], 
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim']
    ).to(device)
    
    # 2. 사전 학습된 뼈대 로드 (전이 학습)
    if os.path.exists(config['paths']['pretrained_weights']):
        encoder.load_pretrained_backbone(config['paths']['pretrained_weights'], freeze=True)
        print("뼈대 가중치 로드 완료")
    
    encoder.eval()

    # 3. 데이터 로드
    data_path = config['paths']['lab_data']
    neural_data = torch.FloatTensor(np.load(os.path.join(data_path, "spikes.npy"))).to(device)
    choices = np.load(os.path.join(data_path, "choices.npy"))
    rewards = np.load(os.path.join(data_path, "rewards.npy"))
    
    # 4. [1단계] Feature Representation 추출
    with torch.no_grad():
        # 인코더를 통과시켜 (Batch, latent_dim) 형태의 피처 추출
        feature_embeddings = encoder(neural_data).cpu().numpy()

    # 5. [2단계] 지도 학습 모드 (Hypothesis-driven)
    # 행동 레이블(k)을 전달하여 해당 레이블을 기준으로 대조 학습 수행
    behavioral_labels = (choices * 2) + rewards
    
    model = cebra.CEBRA(
        output_dimension=config['model']['latent_dim'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        max_iterations=config['training'].get('epochs', 1000),
        device='cuda'
    )

    # fit에 feature_embeddings를 전달하여 정렬
    model.fit(feature_embeddings, behavioral_labels)
    
    # 결과 저장
    save_path = os.path.join(config['paths']['model_save'], "hypothesis_model.pt")
    model.save(save_path)
    print(f"✅ Hypothesis-driven 학습(Fine-tuning) 완료: {save_path}")

if __name__ == "__main__":
    run_hypothesis_training()