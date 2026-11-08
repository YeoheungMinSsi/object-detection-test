from pathlib import Path
import json
import random
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET


DATASET = Path(__file__).resolve().parent / "dataset" / "images-label"


def load_dataset():
    samples = []
    labels = sorted((DATASET / "Annotations").glob("*.xml"))
    if not labels:
        raise ValueError(f"XML 라벨이 없습니다: {DATASET / 'Annotations'}")

    for label_path in labels:
        root = ET.parse(label_path).getroot()
        filename = root.findtext("filename")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"올바르지 않은 이미지 이름: {label_path}")
        image_path = DATASET / "images" / filename
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

    tasks = json.loads((DATASET.parent / "label.json").read_text())
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


if __name__ == "__main__":
    samples = load_dataset()
    train_data, val_data, test_data = split_dataset(samples)
    for name, group in (("학습", train_data), ("검증", val_data), ("테스트", test_data)):
        print(f"\n{name} 데이터: {len(group)}장")
        for sample in group:
            print(f"  {sample['image'].name}: 사람 {len(sample['boxes'])}명")
