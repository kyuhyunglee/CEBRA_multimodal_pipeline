# scripts/train_discovery.py
import cebra
import numpy as np
import yaml
import os
import torch
from models.encoders import ProbeEncoder

def run_discovery_training():
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. 인코더 초기화 및 가중치 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 데이터셋에 맞는 입력 뉴런 수 설정 필요 (config에서 가져옴)
    encoder = ProbeEncoder(input_neurons=config['dataset']['input_neurons'], 
                           latent_dim=config['model']['latent_dim']).to(device)
    
    # Pre-trained backbone 로드 (가중치가 있다면)
    if os.path.exists(config['paths']['pretrained_weights']):
        encoder.load_pretrained_backbone(config['paths']['pretrained_weights'])
    
    encoder.eval() # 학습된 피처 추출 모드

    # 2. 데이터 로드 (학습 데이터)
    data_path = config['paths']['steinmetz_data']
    raw_neural_data = torch.FloatTensor(np.load(os.path.join(data_path, "spikes.npy"))).to(device)

    # 3. [1단계] Encoder를 통한 Feature Representation 추출
    with torch.no_grad():
        # (Batch, Neurons, Time) 형태의 텐서를 인코더에 통과
        feature_embeddings = encoder(raw_neural_data).cpu().numpy()

    # 4. [2단계] 추출된 Embedding에 대해 CEBRA Contrastive Learning 수행
    # 이제 모델은 raw data가 아닌, encoder가 압축한 feature embedding을 학습함
    cebra_model = cebra.CEBRA(
        output_dimension=config['model']['latent_dim'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        max_iterations=config['training'].get('epochs', 1000),
        device='cuda'
    )

    # Discovery-driven: 시간 정보만 사용하여 Contrastive Learning
    cebra_model.fit(feature_embeddings)
    
    # 5. 최종 결과 저장
    save_path = os.path.join(config['paths']['model_save'], "discovery_embedding_model.pt")
    cebra_model.save(save_path)
    print(f" Stage 2 학습(Contrastive Learning) 완료: {save_path}")

if __name__ == "__main__":
    run_discovery_training()