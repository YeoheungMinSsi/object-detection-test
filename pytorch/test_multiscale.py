"""새 다중 해상도 체크포인트의 검출 결과/지표를 확인한다."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import torch

from model.network import create_person_detection
from multiscale.data import load_manifest_samples, read_image, preprocess
from multiscale.ops import DetectionMetrics, postprocess


def draw(image, truth, detection, index, total):
    image = image.copy()
    height, width = image.shape[:2]
    def rectangle(box, color, text):
        x1, y1, x2, y2 = box
        left, top = int(x1*(width-1)), int(y1*(height-1))
        right, bottom = int(x2*(width-1)), int(y2*(height-1))
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, text, (left, max(15, top-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for box in truth:
        rectangle(box, (0, 255, 0), 'GT')
    for box, score in zip(detection['boxes'].tolist(), detection['scores'].tolist()):
        rectangle(box, (0, 0, 255), f'{score:.2f}')
    ratio = min(1200/width, 800/height, 2.0)
    image = cv2.resize(image, (max(1, round(width*ratio)), max(1, round(height*ratio))))
    image = cv2.copyMakeBorder(image, 65, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(image, f'{index+1}/{total} GT(green)={len(truth)} Pred(red)={len(detection["boxes"])}',
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, 'D/Space next | A previous | Q/Esc quit', (8, 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--score', type=float)
    parser.add_argument('--nms-iou', type=float)
    parser.add_argument('--match-iou', type=float)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit는 0 이상이어야 합니다.')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'person_multiscale_v1':
        raise ValueError('train_multiscale.py로 생성한 가중치가 필요합니다. 이전 .pt는 호환되지 않습니다.')
    score = args.score if args.score is not None else checkpoint['score']
    nms_iou = args.nms_iou if args.nms_iou is not None else checkpoint['nms_iou']
    match_iou = args.match_iou if args.match_iou is not None else checkpoint['match_iou']
    if not 0 <= score <= 1 or not 0 <= nms_iou <= 1 or not 0 < match_iou <= 1:
        parser.error('score/nms-iou는 0~1, match-iou는 0 초과 1 이하여야 합니다.')
    image_size = checkpoint['image_size']
    samples, _ = load_manifest_samples(args.weights, args.split, args.dataset)
    if args.limit:
        samples = samples[:args.limit]
    # 수정된 부분(norm 추가 / 26.11.23)
    model = create_person_detection(norm=checkpoint.get('norm', 'bn')).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    output = args.output or (args.weights.resolve().parent/'inspection'/
                            f'{args.split}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    index = 0
    print(f'Device={device} epoch={checkpoint["epoch"]} score={score} Output={output.resolve()}', flush=True)
    try:
        with torch.inference_mode():
            while True:
                sample = samples[index]
                image = read_image(sample['image'])
                raw = model(preprocess(image, image_size).unsqueeze(0).to(device))
                detection = postprocess(raw, image_size, score, nms_iou)[0]
                meter = DetectionMetrics(match_iou)
                meter.update([detection], [{'boxes': torch.tensor(sample['boxes']).reshape(-1, 4)}])
                metrics = meter.result()
                canvas = draw(image, sample['boxes'], detection, index, len(samples))
                ok, encoded = cv2.imencode('.jpg', canvas)
                if not ok:
                    raise RuntimeError('이미지 인코딩 실패')
                encoded.tofile(str(output/f'{index+1:04d}_{sample["image"].stem}.jpg'))
                records[index] = {'image': str(sample['image']), 'gt': len(sample['boxes']),
                                  'predictions': len(detection['boxes']), **metrics}
                with (output/'report.csv').open('w', newline='', encoding='utf-8-sig') as file:
                    writer = csv.DictWriter(file, fieldnames=list(records[index]))
                    writer.writeheader()
                    writer.writerows(records[key] for key in sorted(records))
                tp = sum(r['tp'] for r in records.values())
                fp = sum(r['fp'] for r in records.values())
                fn = sum(r['fn'] for r in records.values())
                summary = dict(evaluated_images=len(records), total_images=len(samples),
                               complete=len(records)==len(samples), tp=tp, fp=fp, fn=fn,
                               precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                               f1=2*tp/max(2*tp+fp+fn, 1), score=score,
                               nms_iou=nms_iou, match_iou=match_iou)
                (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
                print(f'[{index+1}/{len(samples)}] {sample["image"].name} '
                      f'GT={len(sample["boxes"])} Pred={len(detection["boxes"])} '
                      f'TP={metrics["tp"]} FP={metrics["fp"]} FN={metrics["fn"]}', flush=True)
                if args.no_show:
                    if index == len(samples)-1:
                        break
                    index += 1
                    continue
                cv2.imshow('Multiscale person detection', canvas)
                while True:
                    key = cv2.waitKey(100)
                    if cv2.getWindowProperty('Multiscale person detection', cv2.WND_PROP_VISIBLE) < 1:
                        return
                    if key == -1:
                        continue
                    key &= 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        return
                    if key in (ord('d'), ord('D'), 32):
                        if index == len(samples)-1:
                            return
                        index += 1
                        break
                    if key in (ord('a'), ord('A')):
                        index = max(0, index-1)
                        break
    finally:
        if not args.no_show:
            cv2.destroyAllWindows()
        print(f'Results: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
