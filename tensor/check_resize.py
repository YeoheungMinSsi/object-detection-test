from pathlib import Path
import sys

import cv2

from prepare_dataset import load_dataset, prepare_samples


samples = load_dataset()
sample = next((s for s in samples if s["boxes"] and
               (len(sys.argv) == 1 or s["image"].name == sys.argv[1])), None)
if sample is None:
    raise ValueError("사람 라벨이 있는 이미지 이름을 지정하세요.")

image = cv2.imread(str(sample["image"]))
if image is None:
    raise RuntimeError(f"이미지를 읽을 수 없습니다: {sample['image']}")

height, width = image.shape[:2]
resized = cv2.resize(image, (256, 256))
from_original = resized.copy()
from_target = resized.copy()
expected_boxes = []

for x_min, y_min, x_max, y_max in sample["boxes"]:
    # 정규화 좌표를 원본 픽셀로 되돌린 후 256×256으로 크기를 변경한다.
    original_box = (x_min * width, y_min * height, x_max * width, y_max * height)
    scaled_box = (
        original_box[0] * 256 / width,
        original_box[1] * 256 / height,
        original_box[2] * 256 / width,
        original_box[3] * 256 / height,
    )
    expected_boxes.append(tuple(round(value) for value in scaled_box))
    cv2.rectangle(from_original, (round(scaled_box[0]), round(scaled_box[1])),
                  (round(scaled_box[2]), round(scaled_box[3])), (0, 255, 0), 2)

_, targets = prepare_samples([sample])
actual_boxes = []
for row in range(14):
    for column in range(14):
        for slot in range(5):
            score, local_x, local_y, box_width, box_height = targets[0, row, column, slot]
            if score == 0:
                continue
            center_x = (column + local_x) / 14
            center_y = (row + local_y) / 14
            box = (
                (center_x - box_width / 2) * 256,
                (center_y - box_height / 2) * 256,
                (center_x + box_width / 2) * 256,
                (center_y + box_height / 2) * 256,
            )
            actual_boxes.append(tuple(round(value) for value in box))
            cv2.rectangle(from_target, (round(box[0]), round(box[1])),
                          (round(box[2]), round(box[3])), (0, 0, 255), 2)

if sorted(expected_boxes) != sorted(actual_boxes):
    raise AssertionError("크기 변경 후 박스와 학습 정답 박스가 다릅니다.")

output = Path(__file__).resolve().parent / "dataset" / "resize_check.jpg"
if not cv2.imwrite(str(output), cv2.hconcat([from_original, from_target])):
    raise RuntimeError(f"결과 이미지를 저장하지 못했습니다: {output}")
print(f"{sample['image'].name}: 사람 {len(sample['boxes'])}명")
print(f"원본 {width}×{height} → 학습 입력 256×256")
print(f"왼쪽 초록=원본 박스 변환, 오른쪽 빨강=학습 정답 복원: {output}")
