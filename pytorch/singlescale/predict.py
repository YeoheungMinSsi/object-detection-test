"""PyTorch 모델 로딩, 이미지 전처리, 박스 복원 및 중복 제거."""
from pathlib import Path

import cv2
import numpy as np
import torch

from model.network import create_person_detection


def load_model(weights, device):
    checkpoint = torch.load(Path(weights), map_location="cpu", weights_only=True)
    model = create_person_detection()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def read_image(path):
    # np.fromfile/imdecode 조합은 Windows의 한글 경로도 처리한다.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def image_tensor(image, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512)).astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)


def non_maximum_suppression(boxes, scores, iou_threshold, max_detections):
    """점수가 높은 박스를 남기고 IoU가 큰 중복 박스를 제거한다."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() and len(keep) < max_detections:
        current = order[0]
        keep.append(int(current))
        rest = order[1:]
        if not rest.numel():
            break
        top_left = torch.maximum(boxes[current, :2], boxes[rest, :2])
        bottom_right = torch.minimum(boxes[current, 2:], boxes[rest, 2:])
        intersection = (bottom_right - top_left).clamp_min(0).prod(dim=1)
        area_current = (boxes[current, 2:] - boxes[current, :2]).prod()
        area_rest = (boxes[rest, 2:] - boxes[rest, :2]).prod(dim=1)
        iou = intersection / (area_current + area_rest - intersection).clamp_min(1e-8)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long)


def decode_predictions(prediction, score_threshold=0.5, iou_threshold=0.5,
                       max_detections=100):
    # 반환 좌표는 0~1로 정규화한 [xmin, ymin, xmax, ymax]다.
    prediction = prediction.detach().float().cpu()
    if tuple(prediction.shape) != (30, 30, 5, 5):
        raise ValueError(f"예측 형태 오류: {tuple(prediction.shape)}")
    row, column, slot = torch.where(prediction[..., 0] >= score_threshold)
    values = prediction[row, column, slot]
    if not len(values):
        return torch.empty((0, 4)), torch.empty(0)
    center_x = (column + values[:, 1]) / 30
    center_y = (row + values[:, 2]) / 30
    width, height = values[:, 3], values[:, 4]
    boxes = torch.stack((center_x - width / 2, center_y - height / 2,
                         center_x + width / 2, center_y + height / 2), dim=1).clamp(0, 1)
    scores = values[:, 0]
    valid = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    keep = non_maximum_suppression(boxes, scores, iou_threshold, max_detections)
    return boxes[keep], scores[keep]
