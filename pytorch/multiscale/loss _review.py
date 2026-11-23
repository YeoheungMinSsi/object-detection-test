import torch
from torch import nn
from torch.nn import functional as F
from .ops import STRIDES, decode_level
from .targets import TargetAssigner

# NOTE: 모델의 예측과 정답을 비교해, 모델이 얼마나 틀렸는지 계산하는 부분
# 모델은 small, medium, large 3개 값을 출력함

# 박스의 겹침 정도 계산
# IoU대신 GIoU을 이용하는 이유: 두 박스가 전혀 겹치치 않으면 IoU는 모두 0이기 때문
def aligned_giou_loss(boxes, truth):  # boxes: 예측 박스, truth: 정답 박스  / (n, 4) 형태
    # maximum은 두 박스에서 겹치는 영역의 왼쪽 위는 각 좌표의 최댓값
    # minimum은 두 박스에서 겹치는 영역의 오른쪽 아래는 각 좌표의 최솟값
    # ex) 예측 A = [1, 1, 5, 5]  /  # ex) 정답 B = [3, 2, 7, 6]
    # boxes[:, 2:], truth[:, 2:]  ->  뒤 2개의 값을 가지고 옮: [5, 5], [7, 6]
    # boxes[:, :2], truth[:, :2]  ->  앞 2개의 값을 가지고 옮: [1, 1], [3, 2]
    # torch.minimum([5, 5], [7, 6]): 같은 위치에서 작은 값을 가져옴 -> [5, 5] / maximum([1, 1], [3, 2]): 같은 위치에서 큰 값을 가져옴 -> [3, 2]
    # clamp_min(0): 두 박스가 떨어져 있어 너비나 높이가 음수로 계산될 때 0으로 변환(겹치지 않는 영역의 면적은 0이 되야함)
    # [5, 5] - [3, 2] = [2, 3]  ->  prod(-1) 마지막 차원의 두 값(너비와 높이)을 곱함(2 * 3 = 6)
    intersection = (torch.minimum(boxes[:, 2:], truth[:, 2:]) -
                    torch.maximum(boxes[:, :2], truth[:, :2])).clamp_min(0).prod(-1)

    # ex) 예측 A = [1, 1, 5, 5] / 정답 B = [3, 2, 7, 6]
    # [5, 5] - [1, 1] = [4, 4] - prod(-1): 4 * 4 = 16
    # [7, 6] - [3, 2] = [4, 4] - prod(-1): 4 * 4 = 16
    area_a = (boxes[:, 2:]-boxes[:, :2]).clamp_min(0).prod(-1)  # a의 박스 면적
    area_b = (truth[:, 2:]-truth[:, :2]).clamp_min(0).prod(-1)  # b의 박스 면적
    
    # 두 집합 면적의 합 - 교집합(겹치는곳이 2개, 1개만 남기면 됨)
    # ex) 16 + 16 - 6 = 26
    union = (area_a+area_b-intersection).clamp_min(1e-8)  # 두 박스의 합집합 면적, clamp_min(1e-8): 합집합 면적을 최소 0.000000001로 제한
    
    # 두 박스를 모두 포함하는 가장 작은 사각형(C) / 교집합을 구할 때 최솟값,최댓값의 선택이 반대
    # ex) 예측 A = [1, 1, 5, 5] / 정답 B = [3, 2, 7, 6]
    # maximum([5, 5], [7, 6]) -> [7, 6]  /  minimum([1, 1], [3, 2]) -> [1, 1]  /  [7, 6] - [1, 1] = [6, 5]
    # prod(-1): [6, 5] -> 6 * 5 = 30
    enclosure = (torch.maximum(boxes[:, 2:], truth[:, 2:]) -
                 torch.minimum(boxes[:, :2], truth[:, :2])).prod(-1).clamp_min(1e-8)

    # GIoU: IoU - 빈면적의 비율
    # GIoU: 1 - (교집합/합집합 - (두 박스를 모두 포함하는 가장 작은 사각형 - 합집합)/두 박스를 모두 포함하는 가장 작은 사각형)
    # 만약 두 박스가 일치하면 -> IoU = 1, 빈 면적 = 0, GIoU = 1, 손실 = 0
    return 1 - (intersection/union - (enclosure-union)/enclosure)

# 손실 계산 설정(모델의 예측을 정답과 비교, 학습에 사용할 최종 손실을 계산) / 이전에 설정한 aligned giou loss 여기서 호출

class DetectionLoss(nn.Module):
    """존재 BCEWithLogits + 양성 GIoU + 중심/크기 보조 손실.

    존재 손실은 이미지별 양성 평균과 어려운 배경 평균을 합산한다.
    정답 없는 이미지에서도 최소 32개 배경 후보를 학습한다.
    """
    def __init__(self, box_weight=5.0, background_weight=1.0):
        super().__init__()
        # 각 사람을 어느 해상도의 어느 칸에서 예측하도록 학습할지 정하는 객체
        self.assigner = TargetAssigner()  # 각 정답 사람을 어느 해상도의 어느 칸에 배정할지 결정
        self.box_weight = box_weight  # 박스 겹침 손실에 곱할 가중치
        self.background_weight = background_weight  # 배경 존재 손실에 곱할 가중치

    # 정답 배정과 초기 준비
    # outputs = {  / 모델 출력
        # 'small':  ...,  # (B, 5, 64, 64)
        # 'medium': ...,  # (B, 5, 32, 32)
        # 'large':  ...,  # (B, 5, 16, 16)
    # }  / B는 이미지 개수/존재 logit, tx, ty, tw, th/64/64

    # targets = [  / 정답: 이미지별 박스 목록
    #     {
    #         'boxes': torch.tensor([
    #             [0.20, 0.30, 0.30, 0.40]
    #         ]),
    #         'path': 'person.jpg'
    #     }
    # ]
    def forward(self, outputs, targets, image_size):
        # 정답을 담당할 칸 결정(크게 두가지의 해상도별 정보를 반환), 정답 박스는 여러 해상도의 모든 칸에 배정하지 않음
        # TargetAssigner가 담당 칸을 정함
        # assignments[name]['positive']  / (배치, 높이, 너비), bool(정답이 배정된 칸이면 true, 아니면 false)  
        # assignments[name]['boxes']     # (배치, 높이, 너비, 4)
        assignments = self.assigner(outputs, targets, image_size)
        # 0인 손실의 기준값을 만듦, 단순 0이 아닌 모델 출력의 계산 그래프에 연결된 값
        # 정답이 전혀 없어 박스 손실을 계산하지 않는 경우에도 역전파에 사용 가능
        zero = sum(value.sum()*0 for value in outputs.values())
        # 각 해상도의 계산 결과를 모아 둘 리스트
        giou_terms, auxiliary_terms = [], []
        all_logits, all_positive = [], []
        for name, stride in STRIDES.items():
            raw = outputs[name]
            positive = assignments[name]['positive']
            truth = assignments[name]['boxes']
            all_logits.append(raw[:, 0].flatten(1))
            all_positive.append(positive.flatten(1))
            if positive.any():
                decoded = decode_level(raw, stride, image_size)
                giou_terms.append(aligned_giou_loss(decoded[positive], truth[positive]))
                b, row, column = positive.nonzero(as_tuple=True)
                values = raw.permute(0, 2, 3, 1)[positive]
                selected = truth[positive]
                center = (selected[:, :2]+selected[:, 2:])/2 * image_size / stride
                cell = torch.stack((column, row), dim=-1).to(values)
                offsets = center-cell
                sizes = (selected[:, 2:]-selected[:, :2]) * image_size / stride
                center_error = F.smooth_l1_loss(values[:, 1:3].sigmoid(), offsets,
                                               reduction='none').sum(-1)
                # log 크기를 비교해 큰 박스가 보조 손실을 과도하게 지배하지 않게 한다.
                size_error = F.smooth_l1_loss(F.softplus(values[:, 3:5]).clamp_min(1e-6).log(),
                                             sizes.clamp_min(1e-6).log(),
                                             reduction='none').sum(-1)
                auxiliary_terms.append(center_error+size_error)
        logits = torch.cat(all_logits, dim=1)
        positives = torch.cat(all_positive, dim=1)
        errors = F.binary_cross_entropy_with_logits(logits, positives.to(logits.dtype), reduction='none')
        positive_losses, negative_losses = [], []
        for image_errors, mask in zip(errors, positives):
            count = int(mask.sum())
            positive_losses.append(image_errors[mask].mean() if count else image_errors.sum()*0)
            negatives = image_errors[~mask]
            k = min(len(negatives), max(32, 3*count))
            negative_losses.append(negatives.topk(k).values.mean() if k else image_errors.sum()*0)
        person_loss = torch.stack(positive_losses).mean()
        background_loss = torch.stack(negative_losses).mean()
        box_loss = torch.cat(giou_terms).mean() if giou_terms else zero
        auxiliary_loss = torch.cat(auxiliary_terms).mean() if auxiliary_terms else zero
        total = person_loss+self.background_weight*background_loss+self.box_weight*box_loss+0.5*auxiliary_loss
        return dict(total=total, person=person_loss, background=background_loss,
                    box=box_loss, auxiliary=auxiliary_loss,
                    positives=positives.sum().detach())
