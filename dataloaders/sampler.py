import torch

class CEBRASampler:
    def __init__(self, dataset, time_shift=0):
        """
        dataset: IBLSubjectDataset 인스턴스 (get_data()로 리스트 반환)
        time_shift: 행동 레이블을 기준으로 신경 데이터를 앞당길 오프셋 (단위: bin 개수)
        """
        self.neural_list, self.behavior_list = dataset.get_data()
        self.time_shift = time_shift

    def get_cebra_inputs(self):
        shifted_neural_list = []
        shifted_labels_list = []
        
        for neural_data, labels in zip(self.neural_list, self.behavior_list):
            num_samples = len(neural_data) - self.time_shift
            
            # 유효한 샘플 길이가 안 나오면 해당 세션은 스킵
            if num_samples <= 0:
                continue

            # CEBRA offset 모델에 넣기 위해 형태는 (Time, Neurons) 원본 유지
            # Time-shift 적용: 신경 데이터는 과거~현재, 행동 레이블은 미래
            shifted_neural = neural_data[:num_samples]
            shifted_label = labels[self.time_shift : self.time_shift + num_samples]
            
            # 리스트에 numpy 배열 형태로 어펜드 (뉴런 개수가 달라도 무관함)
            shifted_neural_list.append(shifted_neural.numpy())
            shifted_labels_list.append(shifted_label.numpy())
            
        return shifted_neural_list, shifted_labels_list