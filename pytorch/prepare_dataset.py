from pathlib import Path
import json
import random
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET


import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DATASET = Path(__file__).resolve().parent / "dataset" / "images-label"


def load_dataset(dataset=None):
    # 외부 VOC 데이터셋을 지정하면 기존 labels.json을 섞지 않는다.
    dataset_root = Path(dataset).resolve() if dataset is not None else DATASET
    samples = []
    labels = sorted((dataset_root / "Annotations").glob("*.xml"))
    if not labels:
        raise ValueError(f"XML 라벨이 없습니다: {dataset_root / 'Annotations'}")

    for label_path in labels:
        root = ET.parse(label_path).getroot()
        filename = root.findtext("filename")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"올바르지 않은 이미지 이름: {label_path}")
        image_path = dataset_root / "images" / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"이미지가 없습니다: {image_path}")

        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))
        if width <= 0 or height <= 0:
            raise ValueError(f"올바르지 않은 이미지 크기: {label_path}")

        boxes = []
        for obj in root.findall("object"):
            if obj.findtext("name") != "person":
                continue
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"사각형 좌표가 없습니다: {label_path}")
            x_min, y_min, x_max, y_max = (
                float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")
            )
            if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                raise ValueError(f"이미지 범위를 벗어난 사각형: {label_path}")
            boxes.append([x_min / width, y_min / height, x_max / width, y_max / height])

        samples.append({"image": image_path, "label": label_path, "boxes": boxes})

    tasks = (json.loads((DATASET.parent / "labels.json").read_text(encoding="utf-8-sig"))
             if dataset is None else [])
    used_images = {sample["image"] for sample in samples}
    for task in tasks:
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled", False)]
        if not annotations or any(a.get("result") for a in annotations):
            continue
        source = parse_qs(urlparse(task["data"]["image"]).query).get("d", [""])[0]
        filename = Path(source).name
        if not filename:
            raise ValueError(f"빈 라벨의 이미지 이름이 없습니다: 작업 {task['id']}")
        matches = [p for p in (DATASET / "images").iterdir()
                   if p.is_file() and (p.name == filename or p.name.endswith("__" + filename))]
        if len(matches) != 1 or matches[0] in used_images:
            raise ValueError(f"빈 라벨에 해당하는 이미지가 없거나 중복됩니다: {filename}")
        samples.append({"image": matches[0], "label": None, "boxes": []})
        used_images.add(matches[0])

    return samples


def split_dataset(samples):
    # shortcut: 비슷한 촬영 장면이 서로 다른 그룹에 섞일 수 있음, 장면 정보 확보 시 그룹별로 분리
    shuffled = sorted(samples, key=lambda sample: sample["image"].name)
    random.Random(42).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.60)
    val_end = train_end + int(len(shuffled) * 0.25)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def prepare_samples(samples):
    images = []
    boxes = []

    for sample in samples:
        # 이미지 읽기, image와 xml 중 이미지 만을 가지고 옴
        image = cv2.imdecode(np.fromfile(str(sample["image"]), dtype=np.uint8), cv2.IMREAD_COLOR)

        # 이미지 로드여부 확인
        if image is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {sample['image']}")

        # 학습에 사용할 이미지 형태로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = image.astype("float32") / 255.0

        # 30×30 그리드의 각 칸에 후보 5개와 후보별 값 5개를 저장
        target_boxes = np.zeros((30, 30, 5, 5), dtype=np.float32)

        for box in sample["boxes"]:
            x_min, y_min, x_max, y_max = box

            # 그리드 좌표 계산
            # 왼쪽 위, 오른쪽 아래 좌표를 중심/크기로 변환
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # 사람 중심이 속한 30x30 위치
            column = min(int(center_x * 30), 29)
            row = min(int(center_y * 30), 29)

            # 같은 위치의 빈 후보 칸을 찾기
            for slot in range(5):
                if target_boxes[row, column, slot, 0] == 0:  # 존재 점수가 0이면 빈 칸
                    target_boxes[row, column, slot] = [
                        1.0,  # 존재 점수
                        center_x * 30 - column,  # 그리드 내 상대 좌표
                        center_y * 30 - row,  # 그리드 내 상대 좌표
                        width,
                        height,
                    ] 
                    break
            else:
                raise ValueError(
                        f"한 위치에 사람이 5명보다 많습니다: {sample['image']}"
                    )

        images.append(image)
        boxes.append(target_boxes)

    return np.array(images), np.array(boxes)



class PersonDataset(Dataset):
    """경로와 라벨만 보관하고 요청받은 이미지 한 장만 변환한다."""

    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 기존 전처리/정답 생성 함수를 한 장에만 적용한다.
        images, targets = prepare_samples([self.samples[index]])
        # RGB, float32, 0~1. HWC → CHW로 변환한다.
        image = torch.from_numpy(images[0]).permute(2, 0, 1).contiguous()
        target = torch.from_numpy(targets[0])
        return image, target


if __name__ == "__main__":
    groups = split_dataset(load_dataset())
    for name, samples in zip(("학습", "검증", "테스트"), groups):
        people = sum(len(s["boxes"]) for s in samples)
        print(f"{name}: {len(samples)}장, 사람 {people}명", flush=True)
    if groups[0]:
        image, target = PersonDataset(groups[0])[0]
        print("이미지:", tuple(image.shape), "정답:", tuple(target.shape))
