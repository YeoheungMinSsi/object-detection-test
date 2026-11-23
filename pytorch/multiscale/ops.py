import torch
from torch.nn import functional as F

STRIDES = {'small': 8, 'medium': 16, 'large': 32}


def validate_outputs(outputs, image_size):
    if set(outputs) != set(STRIDES):
        raise ValueError('small, medium, large 출력이 필요합니다.')
    batch = None
    for name, stride in STRIDES.items():
        value = outputs[name]
        if value.ndim != 4 or tuple(value.shape[1:]) != (5, image_size//stride, image_size//stride):
            raise ValueError(f'{name} 출력 형태 오류: {tuple(value.shape)}')
        if batch is not None and batch != value.shape[0]:
            raise ValueError('출력별 배치 크기가 다릅니다.')
        batch = value.shape[0]


def decode_level(raw, stride, image_size):
    """B,5,H,W 원시 출력을 B,H,W,4 정규화 xyxy 박스로 복원.

    학습/검증/추론이 동일한 복원 규칙을 사용한다. 학습 기울기를 유지하므로
    여기서는 detach, 좌표 clamp, NMS를 적용하지 않는다.
    """
    _, _, height, width = raw.shape
    row, column = torch.meshgrid(torch.arange(height, device=raw.device),
                                 torch.arange(width, device=raw.device), indexing='ij')
    center_x = (raw[:, 1].sigmoid() + column) * stride / image_size
    center_y = (raw[:, 2].sigmoid() + row) * stride / image_size
    box_width = F.softplus(raw[:, 3]) * stride / image_size
    box_height = F.softplus(raw[:, 4]) * stride / image_size
    return torch.stack((center_x-box_width/2, center_y-box_height/2,
                        center_x+box_width/2, center_y+box_height/2), dim=-1)


def pairwise_iou(boxes, truth):
    intersection = (torch.minimum(boxes[:, None, 2:], truth[None, :, 2:]) -
                    torch.maximum(boxes[:, None, :2], truth[None, :, :2])).clamp_min(0).prod(-1)
    area_a = (boxes[:, 2:]-boxes[:, :2]).clamp_min(0).prod(-1)
    area_b = (truth[:, 2:]-truth[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area_a[:, None]+area_b[None, :]-intersection).clamp_min(1e-8)


def nms(boxes, scores, threshold, maximum):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() and len(keep) < maximum:
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        if not rest.numel():
            break
        overlap = pairwise_iou(boxes[current:current+1], boxes[rest])[0]
        order = rest[overlap <= threshold]
    return torch.tensor(keep, dtype=torch.long)


@torch.no_grad()
def postprocess(outputs, image_size, score=0.5, nms_iou=0.5, maximum=100):
    validate_outputs(outputs, image_size)
    all_boxes, all_scores = [], []
    for name, stride in STRIDES.items():
        raw = outputs[name].detach().float()
        all_boxes.append(decode_level(raw, stride, image_size).flatten(1, 2))
        all_scores.append(raw[:, 0].sigmoid().flatten(1))
    batch_boxes = torch.cat(all_boxes, dim=1).cpu()
    batch_scores = torch.cat(all_scores, dim=1).cpu()
    results = []
    for boxes, scores in zip(batch_boxes, batch_scores):
        boxes = boxes.clamp(0, 1)
        valid = (scores >= score) & torch.isfinite(scores) & torch.isfinite(boxes).all(-1)
        valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        # NMS 입력을 제한해 매우 낮은 score에서도 계산량을 제어한다.
        if len(scores) > 1000:
            indices = scores.topk(1000).indices
            boxes, scores = boxes[indices], scores[indices]
        keep = nms(boxes, scores, nms_iou, maximum)
        results.append({'boxes': boxes[keep], 'scores': scores[keep]})
    return results


class DetectionMetrics:
    """고정 score/NMS/IoU에서 전체 이미지의 TP/FP/FN을 합산. mAP는 아님."""
    def __init__(self, match_iou=0.5):
        self.match_iou = match_iou
        self.tp = self.fp = self.fn = 0

    def update(self, detections, targets):
        for detection, target in zip(detections, targets):
            boxes, scores = detection['boxes'].cpu(), detection['scores'].cpu()
            truth = target['boxes'].detach().cpu()
            matched = torch.zeros(len(truth), dtype=torch.bool)
            tp = 0
            overlaps = pairwise_iou(boxes, truth)
            for index in scores.argsort(descending=True):
                if not len(truth):
                    break
                ious = overlaps[index].clone()
                ious[matched] = -1
                best = int(ious.argmax())
                if ious[best] >= self.match_iou:
                    matched[best] = True
                    tp += 1
            self.tp += tp
            self.fp += len(boxes)-tp
            self.fn += len(truth)-tp

    def result(self):
        return dict(tp=self.tp, fp=self.fp, fn=self.fn,
                    precision=self.tp/max(self.tp+self.fp, 1),
                    recall=self.tp/max(self.tp+self.fn, 1),
                    f1=2*self.tp/max(2*self.tp+self.fp+self.fn, 1))
