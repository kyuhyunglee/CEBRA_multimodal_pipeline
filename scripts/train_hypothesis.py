# scripts/train_hypothesis.py
import cebra
import numpy as np
import yaml
import os

def run_hypothesis_training():
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 데이터 로드
    data_path = config['paths']['steinmetz_data']
    neural_data = np.load(os.path.join(data_path, "spikes.npy"))
    choices = np.load(os.path.join(data_path, "choices.npy")) # discrete behaviour (k)
    rewards = np.load(os.path.join(data_path, "rewards.npy"))
    
    # 행동 라벨 결합 (discrete behavior labels)
    # 논문 예시: model.fit(s, k) 또는 model.fit(s, c, k)
    behavioral_labels = (choices * 2) + rewards

    model = cebra.CEBRA(
        output_dimension=config['model']['latent_dim'],
        num_hidden_units=config['model']['hidden_dim'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        max_iterations=config['training'].get('epochs', 1000),
        device='cuda' if cebra.utils.is_gpu_available() else 'cpu'
    )

    # 2. Hypothesis-driven mode: 신경 데이터(s)와 행동 라벨(k) 전달
    # CEBRA가 행동 라벨을 기준으로 contrastive learning 수행
    model.fit(neural_data, behavioral_labels)
    
    # 결과 저장
    save_path = os.path.join(config['paths']['pretrained_weights'], "hypothesis_model.pt")
    model.save(save_path)
    print(f"✅ Hypothesis-driven 학습 완료: {save_path}")

if __name__ == "__main__":
    run_hypothesis_training()