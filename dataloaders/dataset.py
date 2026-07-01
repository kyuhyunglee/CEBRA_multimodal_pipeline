# dataloaders/dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SteinmetzCEBRADataset(Dataset):
    """
    Steinmetz 데이터셋을 CEBRA 1D CNN 인코더에 맞게 전처리하는 Dataset 클래스.
    """
    def __init__(self, data_path, time_window=40, time_shift_bins=10):
        super().__init__()
        self.time_window = time_window
        self.time_shift = time_shift_bins
        
        try:
            self.spikes = np.load(os.path.join(data_path, "spikes.npy"))
            self.choices = np.load(os.path.join(data_path, "choices.npy")) 
            self.rewards = np.load(os.path.join(data_path, "rewards.npy"))
        except FileNotFoundError:
            # 로컬 디버깅용 더미 데이터
            self.spikes = np.random.randn(10000, 250).astype(np.float32)
            self.choices = np.random.randint(0, 2, 10000)
            self.rewards = np.random.randint(0, 2, 10000)
        
        max_valid_idx = len(self.spikes) - self.time_window - self.time_shift
        self.valid_indices = np.arange(max_valid_idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        
        # 신경 데이터 (Batch, Neurons, Time) 형태를 위해 Transpose
        spike_window = self.spikes[valid_idx : valid_idx + self.time_window, :]
        spike_tensor = torch.tensor(spike_window, dtype=torch.float32).T 
        
        # 다중 행동 레이블 추출 (Causal Sampling)
        target_idx = valid_idx + self.time_window + self.time_shift
        choice = self.choices[target_idx]
        reward = self.rewards[target_idx]
        multi_label = int((choice * 2) + reward)
        
        return spike_tensor, torch.tensor(multi_label, dtype=torch.long)