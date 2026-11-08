from pathlib import Path
import xml.etree.ElementTree as ET

import cv2


dataset = Path(__file__).resolve().parent / "dataset" / "images-label"
label_files = sorted((dataset / "Annotations").glob("*.xml"))

if not label_files:
    raise RuntimeError(f"XML 라벨을 찾을 수 없습니다: {dataset / 'Annotations'}")

print("d: 다음 이미지 | a: 이전 이미지 | q: 종료")
index = 0

try:
    while True:
        label_file = label_files[index]
        root = ET.parse(label_file).getroot()
        filename = root.findtext("filename")
        if not filename:
            raise ValueError(f"이미지 이름이 없습니다: {label_file}")

        image_path = dataset / "images" / filename
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {image_path}")

        for obj in root.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"사각형 좌표가 없습니다: {label_file}")
            x_min = int(float(box.findtext("xmin")))
            y_min = int(float(box.findtext("ymin")))
            x_max = int(float(box.findtext("xmax")))
            y_max = int(float(box.findtext("ymax")))
            name = obj.findtext("name", "unknown")

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, name, (x_min, max(y_min - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"[{index + 1}/{len(label_files)}] {filename}")
        cv2.imshow("Check labels", image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("d") and index < len(label_files) - 1:
            index += 1
        elif key == ord("a") and index > 0:
            index -= 1
finally:
    cv2.destroyAllWindows()
