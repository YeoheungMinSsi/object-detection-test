import torch
from .ops import STRIDES, validate_outputs


class TargetAssigner:
    """크기로 우선 해상도를 정하고 중심 칸 하나를 양성으로 배정한다.

    512 입력 기준 긴 변 64/160픽셀을 경계로 사용한다. 충돌하면 다른 해상도의
    중심 칸을 시도한다. 모두 차 있으면 라벨을 버리지 않고 오류를 낸다.
    이는 교육용 정적 배정이며 특정 YOLO의 동적 배정 구현이 아니다.
    """
    def __call__(self, outputs, targets, image_size):
        validate_outputs(outputs, image_size)
        batch = outputs['small'].shape[0]
        if len(targets) != batch:
            raise ValueError('정답 수와 배치 크기가 다릅니다.')
        assigned = {}
        for name, raw in outputs.items():
            _, _, height, width = raw.shape
            assigned[name] = {
                'positive': torch.zeros((batch, height, width), dtype=torch.bool, device=raw.device),
                'boxes': raw.new_zeros((batch, height, width, 4)),
            }
        names = tuple(STRIDES)
        for image_index, target in enumerate(targets):
            boxes = target['boxes'].to(outputs['small'])
            if boxes.ndim != 2 or boxes.shape[-1] != 4:
                raise ValueError('정답 박스는 (사람 수, 4) xyxy여야 합니다.')
            valid = torch.isfinite(boxes).all() and (boxes >= 0).all() and (boxes <= 1).all()
            if not valid or not ((boxes[:, 2:] - boxes[:, :2]) > 0).all():
                raise ValueError(f'잘못된 정답 좌표: {target.get("path", image_index)}')
            # 작은 사람을 먼저 배정해 촘촘한 해상도의 위치를 확보한다.
            areas = (boxes[:, 2:]-boxes[:, :2]).prod(-1)
            for index in areas.argsort():
                box = boxes[index]
                center = (box[:2]+box[2:])/2
                long_side = float((box[2:]-box[:2]).max())
                preferred = 0 if long_side <= 0.125 else (1 if long_side <= 0.3125 else 2)
                order = sorted(range(3), key=lambda value: abs(value-preferred))
                for level in order:
                    name = names[level]
                    height, width = outputs[name].shape[-2:]
                    column = min(int(center[0]*width), width-1)
                    row = min(int(center[1]*height), height-1)
                    destination = assigned[name]
                    if not destination['positive'][image_index, row, column]:
                        destination['positive'][image_index, row, column] = True
                        destination['boxes'][image_index, row, column] = box
                        break
                else:
                    raise ValueError(f'모든 해상도에서 중심 칸 충돌: {target.get("path", image_index)}')
        return assigned
