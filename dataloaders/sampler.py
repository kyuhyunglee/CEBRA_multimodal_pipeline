# dataloaders/sampler.py
from torch.utils.data import DataLoader
from .dataset import SteinmetzCEBRADataset

def get_dataloaders(config):
    """
    configs/default_config.yaml의 설정을 바탕으로 Train DataLoader를 생성합니다.
    """
    data_path = config['paths']['steinmetz_data']
    time_window = config['data'].get('time_window', 40)
    time_shift = config['data'].get('time_shift_ms', 10) # 10 bins
    batch_size = config['training']['batch_size']
    
    dataset = SteinmetzCEBRADataset(
        data_path=data_path,
        time_window=time_window,
        time_shift_bins=time_shift
    )
    
    # 향후 PyTorch의 custom Sampler를 구현한다면 이 부분에 적용합니다.
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,      # 랜덤하게 데이터를 섞어 InfoNCE Loss 성능 극대화
        drop_last=True     # 일정한 Batch 크기 유지
    )
    
    return dataloader