import os
import glob
import numpy as np
import pandas as pd
import torch

class IBLSubjectDataset:
    def __init__(self, subject_path, bin_size=0.025):
        """
        subject_path: 특정 개체의 최상위 폴더 경로 (예: './data/Subjects/CSHL045')
        bin_size: 스파이크를 묶을 시간 단위 (초)
        """
        self.subject_path = subject_path
        self.bin_size = bin_size
        
        # 다중 세션을 담을 리스트 초기화
        self.sessions_neural = []
        self.sessions_behavior = []

        # 하위의 모든 alf 폴더 탐색 (모든 세션 가져오기)
        alf_paths = glob.glob(os.path.join(self.subject_path, '**', 'alf'), recursive=True)
        if not alf_paths:
            raise FileNotFoundError(f"'{self.subject_path}' 하위에 alf 디렉토리를 찾을 수 없습니다.")
            
        for alf_dir in alf_paths:
            self._prepare_session_data(alf_dir)

    def _prepare_session_data(self, alf_dir):
        spike_times_path = os.path.join(alf_dir, 'spikes.times.npy')
        spike_clusters_path = os.path.join(alf_dir, 'spikes.clusters.npy')
        trials_path = os.path.join(alf_dir, '_ibl_trials.table.pqt')
        
        # 파일이 하나라도 누락된 세션은 스킵
        if not (os.path.exists(spike_times_path) and os.path.exists(trials_path) and os.path.exists(spike_clusters_path)):
            return

        spike_times = np.load(spike_times_path)
        spike_clusters = np.load(spike_clusters_path)
        trials_df = pd.read_parquet(trials_path)
        
        # 1. 스파이크 비닝
        max_time = np.max(spike_times)
        time_bins = np.arange(0, max_time + self.bin_size, self.bin_size)
        num_clusters = np.max(spike_clusters) + 1
        binned_spikes = np.zeros((len(time_bins) - 1, num_clusters))
        
        for cluster_id in np.unique(spike_clusters):
            cluster_spike_times = spike_times[spike_clusters == cluster_id]
            counts, _ = np.histogram(cluster_spike_times, bins=time_bins)
            binned_spikes[:, cluster_id] = counts
            
        # 2. 다중 행동 레이블 정렬 (Choice, Feedback)
        choices = np.zeros(len(time_bins) - 1)
        rewards = np.zeros(len(time_bins) - 1)
        
        for _, trial in trials_df.iterrows():
            start_time = trial.get('stimOn_times', np.nan)
            end_time = trial.get('feedback_times', np.nan)
            
            if pd.isna(start_time) or pd.isna(end_time): 
                continue
                
            start_idx = int(start_time / self.bin_size)
            end_idx = int(end_time / self.bin_size)
            
            if start_idx < len(choices) and end_idx < len(choices):
                choices[start_idx:end_idx] = trial.get('choice', 0)
                rewards[start_idx:end_idx] = trial.get('feedbackType', 0)
                
        # 각 세션의 데이터를 리스트에 추가
        self.sessions_neural.append(torch.FloatTensor(binned_spikes))
        self.sessions_behavior.append(torch.FloatTensor(np.stack([choices, rewards], axis=1)))

    def get_data(self):
        # 여러 세션이 담긴 리스트 반환
        return self.sessions_neural, self.sessions_behavior