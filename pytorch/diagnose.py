"""좌표 왕복 검사 및 소수 이미지 암기 검사. 일반화 성능 평가용이 아니다."""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from model.network import create_person_detection
from model.losses import detection_loss
from prepare_dataset import load_dataset, PersonDataset
from predict import decode_predictions, read_image
from test_model import annotate, loss_parts
from train import save_checkpoint


def save_image(path, image):
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError(f'이미지 인코딩 실패: {path}')
    encoded.tofile(str(path))


def coordinates(samples, output):
    dataset = PersonDataset(samples)
    maximum_error = 0.0
    with (output / 'coordinates.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(('image', 'gt_count', 'decoded_count', 'max_coordinate_error'))
        for index, sample in enumerate(samples):
            _, target = dataset[index]
            # IoU=1로 중복 제거를 비활성화해 겹친 정답도 모두 비교한다.
            boxes, scores = decode_predictions(target, 0.5, 1.0, 4500)
            expected = torch.tensor(sample['boxes'], dtype=torch.float32).reshape(-1, 4)
            if len(boxes) != len(expected):
                raise AssertionError(f'정답 개수 불일치: {sample["image"]}')
            remaining = boxes.clone()
            error = 0.0
            for box in expected:
                distances = (remaining - box).abs().amax(dim=1)
                match = int(distances.argmin())
                error = max(error, float(distances[match]))
                remaining = torch.cat((remaining[:match], remaining[match+1:]))
            if error > 1e-5:
                raise AssertionError(f'좌표 복원 오류: {sample["image"]}, error={error}')
            maximum_error = max(maximum_error, error)
            writer.writerow((sample['image'].name, len(expected), len(boxes), error))
            # 처음 10장은 초록 원본 정답 위에 빨간 복원 박스를 겹쳐 저장한다.
            if index < 10:
                canvas = annotate(read_image(sample['image']), sample, boxes, scores,
                                  index, len(samples))
                save_image(output / f'{index+1:04d}_coordinates.jpg', canvas)
    print(f'PASS: {len(samples)} images, max normalized coordinate error={maximum_error:.8g}', flush=True)


def match_counts(boxes, scores, ground_truth, iou_threshold=0.5):
    # 점수순으로 정답과 일대일 매칭. 같은 사람을 여러 번 맞춰도 TP는 한 번이다.
    truth = torch.tensor(ground_truth, dtype=torch.float32).reshape(-1, 4)
    matched = torch.zeros(len(truth), dtype=torch.bool)
    true_positive = 0
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
            true_positive += 1
    return true_positive, len(boxes)-true_positive, len(truth)-true_positive


def memorize(samples, output, args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    dataset = PersonDataset(samples)
    # 진단용 소수 이미지만 RAM에 보관한다(최대 5장).
    data = [dataset[index] for index in range(len(dataset))]
    model = create_person_detection().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-7)
    names = [str(sample['image']) for sample in samples]
    (output / 'split.json').write_text(json.dumps(
        {'train': names, 'val': names, 'test': []}, indent=2), encoding='utf-8')
    (output / 'config.json').write_text(json.dumps({
        'dataset': str(args.dataset.resolve()), 'diagnostic_only': True,
        'note': 'Train and val intentionally use identical images. Not generalization evaluation.',
        'epochs': args.epochs, 'lr': args.lr,
    }, indent=2), encoding='utf-8')
    best = float('inf')
    with (output / 'history.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(('epoch', 'eval_loss', 'person_loss', 'background_loss',
                         'box_loss', 'tp', 'fp', 'fn', 'precision', 'recall'))
        for epoch in range(1, args.epochs+1):
            model.train()
            for index in torch.randperm(len(data)).tolist():
                image, target = data[index]
                image, target = image.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = detection_loss(target, model(image))
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Non-finite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            # 동일한 이미지를 dropout 없이 평가한다.
            model.eval()
            totals = np.zeros(3)
            tp = fp = fn = 0
            with torch.inference_mode():
                for index, (image, target) in enumerate(data):
                    prediction = model(image.unsqueeze(0).to(device))[0]
                    totals += loss_parts(target.to(device), prediction)
                    boxes, scores = decode_predictions(prediction, args.score, 0.5, 4500)
                    counts = match_counts(boxes, scores, samples[index]['boxes'])
                    tp += counts[0]
                    fp += counts[1]
                    fn += counts[2]
            means = totals / len(data)
            evaluation_loss = float(means.sum())
            precision = tp / max(tp+fp, 1)
            recall = tp / max(tp+fn, 1)
            writer.writerow((epoch, evaluation_loss, *means, tp, fp, fn, precision, recall))
            file.flush()
            if evaluation_loss < best:
                best = evaluation_loss
                save_checkpoint(output / 'best.pt', model, optimizer, epoch, evaluation_loss)
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                save_checkpoint(output / 'last.pt', model, optimizer, epoch, evaluation_loss)
                print(f'Epoch {epoch}/{args.epochs}: loss={evaluation_loss:.5f} '
                      f'TP={tp} FP={fp} FN={fn} P={precision:.3f} R={recall:.3f}', flush=True)
    print('Diagnostic training completed. Evaluate both best.pt and last.pt.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('coordinates', 'memorize'))
    parser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parent / 'dataset/new_images')
    parser.add_argument('--count', type=int, default=1, help='암기 검사 이미지 수: 1~5')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--score', type=float, default=0.5)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 5 or args.epochs < 1:
        parser.error('count는 1~5, epochs는 1 이상이어야 합니다.')
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.score <= 1:
        parser.error('lr는 양수, score는 0~1이어야 합니다.')
    torch.manual_seed(42)
    samples = load_dataset(args.dataset)
    output = args.output or (Path(__file__).resolve().parent / 'diagnostics' /
                            f'{args.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    output.mkdir(parents=True, exist_ok=False)
    print('Output:', output, flush=True)
    if args.mode == 'coordinates':
        coordinates(samples, output)
    else:
        selected = [sample for sample in samples if sample['boxes']][:args.count]
        if len(selected) < args.count:
            raise ValueError('양성 이미지가 부족합니다.')
        print('Selected:', [s['image'].name for s in selected], flush=True)
        memorize(selected, output, args)


if __name__ == '__main__':
    main()
