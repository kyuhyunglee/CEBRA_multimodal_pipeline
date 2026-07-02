# scripts/train_discovery.py
import cebra
import numpy as np
import yaml
import os

def run_discovery_training():
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 데이터 로드 (numpy 형식)
    data_path = config['paths']['steinmetz_data']
    neural_data = np.load(os.path.join(data_path, "spikes.npy"))

    # CEBRA 모델 정의
    model = cebra.CEBRA(
        output_dimension=config['model']['latent_dim'],
        num_hidden_units=config['model']['hidden_dim'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        max_iterations=config['training'].get('epochs', 1000),
        device='cuda_if_available'
    )

    # 1. Discovery-driven mode: 오직 신경 데이터(s)만 전달
    # CEBRA가 자동으로 시간적 근접성을 기반으로 contrastive learning 수행
    model.fit(neural_data)
    
    # 결과 저장
    save_path = os.path.join(config['paths']['pretrained_weights'], "discovery_model.pt")
    model.save(save_path)
    print(f"✅ Discovery-driven 학습 완료: {save_path}")

if __name__ == "__main__":
    run_discovery_training()