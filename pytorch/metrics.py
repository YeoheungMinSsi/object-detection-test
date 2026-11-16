"""고정 score/NMS/매칭 IoU에서 이미지별 일대일 매칭 검출 지표."""
import torch
from predict import decode_predictions


def match_counts(boxes, scores, truth, iou_threshold=0.5):
    boxes, scores = boxes.detach().cpu(), scores.detach().cpu()
    truth = torch.as_tensor(truth, dtype=torch.float32).reshape(-1, 4).cpu()
    matched = torch.zeros(len(truth), dtype=torch.bool)
    tp = 0
    for box in boxes[scores.argsort(descending=True)]:
        if not len(truth):
            break
        intersection = (torch.minimum(box[2:], truth[:, 2:]) -
                        torch.maximum(box[:2], truth[:, :2])).clamp_min(0).prod(dim=1)
        area = (box[2:] - box[:2]).prod()
        truth_area = (truth[:, 2:] - truth[:, :2]).prod(dim=1)
        ious = intersection / (area + truth_area - intersection).clamp_min(1e-8)
        ious[matched] = -1
        best = int(ious.argmax())
        if ious[best] >= iou_threshold:
            matched[best] = True
            tp += 1
    return tp, len(boxes)-tp, len(truth)-tp


class DetectionMetrics:
    def __init__(self, score=0.5, nms_iou=0.5, match_iou=0.5, max_detections=100):
        self.score = score
        self.nms_iou = nms_iou
        self.match_iou = match_iou
        self.max_detections = max_detections
        self.tp = self.fp = self.fn = 0

    def update(self, predictions, targets):
        for prediction, target in zip(predictions, targets):
            boxes, scores = decode_predictions(prediction, self.score, self.nms_iou,
                                                self.max_detections)
            # 정답에는 NMS를 적용하지 않아 겹친 사람도 보존한다.
            truth, _ = decode_predictions(target, 0.5, 1.0, 4500)
            tp, fp, fn = match_counts(boxes, scores, truth, self.match_iou)
            self.tp += tp
            self.fp += fp
            self.fn += fn

    def result(self):
        precision = self.tp / max(self.tp+self.fp, 1)
        recall = self.tp / max(self.tp+self.fn, 1)
        f1 = 2*self.tp / max(2*self.tp+self.fp+self.fn, 1)
        return dict(tp=self.tp, fp=self.fp, fn=self.fn,
                    precision=precision, recall=recall, f1=f1)
