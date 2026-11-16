"""best.pt와 같은 폴더의 split.json을 사용해 실제 검증/테스트 이미지를 확인한다."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from prepare_dataset import load_dataset, prepare_samples
from predict import load_model, read_image, image_tensor, decode_predictions


def loss_parts(target, prediction):
    # 학습 손실과 같은 계산을 세 항으로 나눠 진단한다(이미지 한 장 기준).
    person = target[..., 0]
    background = 1 - person
    error = F.binary_cross_entropy(prediction[..., 0].clamp(1e-7, 1-1e-7),
                                   person, reduction="none")
    person_loss = (error * person).sum() / person.sum().clamp_min(1)
    background_loss = (error * background).sum() / background.sum().clamp_min(1)
    box_error = (target[..., 1:] - prediction[..., 1:]).square().sum(dim=-1)
    box_loss = (box_error * person).sum() / person.sum().clamp_min(1)
    return tuple(float(value) for value in (person_loss, background_loss, box_loss))


def select_samples(weights, split, dataset=None):
    manifest_path = weights.parent / "split.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"학습 당시의 이미지 분리 기록이 필요합니다: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    # 학습 config에서 데이터셋 경로를 자동 복원한다.
    config_path = weights.parent / "config.json"
    if dataset is None and config_path.is_file():
        dataset = json.loads(config_path.read_text(encoding="utf-8-sig")).get("dataset")
    samples = load_dataset(dataset)
    by_path = {str(s["image"].resolve()).casefold(): s for s in samples}
    by_name = {}
    for sample in samples:
        by_name.setdefault(sample["image"].name, []).append(sample)
    selected = []
    for stored_path in manifest[split]:
        sample = by_path.get(str(Path(stored_path).resolve()).casefold())
        if sample is None:
            # 폴더를 옮긴 경우에도 고유한 파일명일 때만 연결한다.
            name = stored_path.replace("\\", "/").rsplit("/", 1)[-1]
            matches = by_name.get(name, [])
            if len(matches) != 1:
                raise ValueError(f"분리 기록의 이미지를 고유하게 찾을 수 없습니다: {stored_path}")
            sample = matches[0]
        selected.append(sample)
    if not selected:
        raise ValueError(f"{split} 그룹에 이미지가 없습니다.")
    return selected


def annotate(image, sample, boxes, scores, index, count):
    canvas = image.copy()
    height, width = canvas.shape[:2]

    def rectangle(box, color, label):
        x1, y1, x2, y2 = box
        left, top = int(x1 * (width-1)), int(y1 * (height-1))
        right, bottom = int(x2 * (width-1)), int(y2 * (height-1))
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        cv2.putText(canvas, label, (left, max(top-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for box in sample["boxes"]:
        rectangle(box, (0, 255, 0), "GT")
    for box, score in zip(boxes.tolist(), scores.tolist()):
        rectangle(box, (0, 0, 255), f"{score:.2f}")
    # 작은 원본 이미지에도 설명이 들어가도록 표시용 크기를 조정한다.
    scale = min(1200 / width, 800 / height, 2.0)
    canvas = cv2.resize(canvas, (max(1, int(width*scale)), max(1, int(height*scale))))
    banner = np.zeros((70, canvas.shape[1], 3), dtype=np.uint8)
    lines = (f"{index+1}/{count} GT(green): {len(sample['boxes'])} Pred(red): {len(boxes)}",
             "D/Space: next | A: previous | Q/Esc: quit")
    for y, line in zip((25, 53), lines):
        cv2.putText(banner, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack((banner, canvas))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="창 없이 결과 이미지만 저장")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path, help="외부 VOC 폴더; 기본값은 학습 설정에서 복원")
    args = parser.parse_args()
    if not (0 <= args.score <= 1 and 0 <= args.iou <= 1):
        parser.error("score/iou는 0~1 범위여야 합니다.")
    if args.limit < 0 or args.max_detections < 1:
        parser.error("limit는 0 이상, max-detections는 1 이상이어야 합니다.")
    weights = args.weights.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    samples = select_samples(weights, args.split, args.dataset)
    if args.limit:
        samples = samples[:args.limit]
    model, checkpoint = load_model(weights, device)
    output = args.output or (weights.parent / "inspection" /
                            f"{args.split}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    print(f"장치: {device} | 저장 에포크: {checkpoint.get('epoch')}", flush=True)
    print(f"저장 당시 val_loss: {checkpoint.get('val_loss')}", flush=True)
    print(f"결과 폴더: {output}", flush=True)
    records = {}
    index = 0
    try:
        with torch.inference_mode():
            while True:
                sample = samples[index]
                image = read_image(sample["image"])
                prediction = model(image_tensor(image, device))[0]
                # 정답 인코딩은 학습과 같은 함수를 사용한다.
                _, targets = prepare_samples([sample])
                target = torch.from_numpy(targets[0]).to(device)
                parts = loss_parts(target, prediction)
                boxes, scores = decode_predictions(prediction, args.score, args.iou,
                                                    args.max_detections)
                canvas = annotate(image, sample, boxes, scores, index, len(samples))
                image_path = output / f"{index+1:04d}_{sample['image'].stem}.jpg"
                ok, encoded = cv2.imencode(".jpg", canvas)
                if not ok:
                    raise RuntimeError("결과 이미지 인코딩 실패")
                encoded.tofile(str(image_path))
                records[index] = (index+1, sample["image"].name, len(sample["boxes"]),
                                  len(boxes), *parts, sum(parts))
                print(f"[{index+1}/{len(samples)}] {sample['image'].name} "
                      f"정답={len(sample['boxes'])} 예측={len(boxes)} | "
                      f"person={parts[0]:.4f} background={parts[1]:.4f} box={parts[2]:.4f}",
                      flush=True)
                # 중간 종료돼도 확인한 이미지의 진단 기록은 남긴다.
                with (output / "report.csv").open("w", newline="", encoding="utf-8-sig") as file:
                    writer = csv.writer(file)
                    writer.writerow(("index", "image", "gt_count", "prediction_count",
                                     "person_loss", "background_loss", "box_loss", "total_loss"))
                    writer.writerows(records[key] for key in sorted(records))
                if args.no_show:
                    if index == len(samples)-1:
                        break
                    index += 1
                    continue
                cv2.imshow("Person detector inspection", canvas)
                while True:
                    key = cv2.waitKey(100)
                    if cv2.getWindowProperty("Person detector inspection", cv2.WND_PROP_VISIBLE) < 1:
                        return
                    if key == -1:
                        continue
                    key &= 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        return
                    if key in (ord("d"), ord("D"), 32):
                        if index == len(samples)-1:
                            return
                        index += 1
                        break
                    if key in (ord("a"), ord("A")):
                        index = max(0, index-1)
                        break
    finally:
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
