import torch
from torch import nn
from torch.nn import functional as F
from .ops import STRIDES, decode_level
from .targets import TargetAssigner

# NOTE: 모델의 예측과 정답을 비교해, 모델이 얼마나 틀렸는지 계산하는 부분
# 모델은 small, medium, large 3개 값을 출력함

# 박스의 겹침 정도 계산
def aligned_giou_loss(boxes, truth):  # boxes: 예측 박스, truth: 정답 박스  / (n, 4) 형태
    intersection = (torch.minimum(boxes[:, 2:], truth[:, 2:]) -
                    torch.maximum(boxes[:, :2], truth[:, :2])).clamp_min(0).prod(-1)

    area_a = (boxes[:, 2:]-boxes[:, :2]).clamp_min(0).prod(-1)  # a의 박스 면적
    area_b = (truth[:, 2:]-truth[:, :2]).clamp_min(0).prod(-1)  # b의 박스 면적
    
    # 두 집합 면적의 합 - 교집합(겹치는곳이 2개, 1개만 남기면 됨)
    union = (area_a+area_b-intersection).clamp_min(1e-8)
    
    # 두 박스를 모두 포함하는 가장 작은 사각형(C) / 교집합을 구할 때 최솟값,최댓값의 선택이 반대
    enclosure = (torch.maximum(boxes[:, 2:], truth[:, 2:]) -
                 torch.minimum(boxes[:, :2], truth[:, :2])).prod(-1).clamp_min(1e-8)

    # GIoU: IoU - 빈면적의 비율
    return 1 - (intersection/union - (enclosure-union)/enclosure)


# 손실 계산 설정
class DetectionLoss(nn.Module):
    """존재 BCEWithLogits + 양성 GIoU + 중심/크기 보조 손실.

    존재 손실은 이미지별 양성 평균과 어려운 배경 평균을 합산한다.
    정답 없는 이미지에서도 최소 32개 배경 후보를 학습한다.
    """
    def __init__(self, box_weight=5.0, background_weight=1.0):
        super().__init__()
        self.assigner = TargetAssigner()  
        self.box_weight = box_weight
        self.background_weight = background_weight

    def forward(self, outputs, targets, image_size):
        assignments = self.assigner(outputs, targets, image_size)
        zero = sum(value.sum()*0 for value in outputs.values())
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
