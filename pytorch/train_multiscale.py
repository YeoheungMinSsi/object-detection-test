"""사용자 정의 3해상도 사람 검출 학습. 기존 train.py는 사용하지 않는다."""
import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.network import create_person_detection
from multiscale.data import load_dataset, split_dataset, PersonDataset, collate_samples, move_targets
from multiscale.loss import DetectionLoss
from multiscale.ops import DetectionMetrics, STRIDES, postprocess

FORMAT = 'person_multiscale_v1'


def save_checkpoint(path, model, optimizer, epoch, losses, metrics, args):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(format=FORMAT, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), epoch=epoch,
                    image_size=args.image_size, strides=STRIDES,
                    decode='sigmoid_xy_softplus_wh', val_losses=losses,
                    metrics=metrics, score=args.score, nms_iou=args.nms_iou,
                    match_iou=args.match_iou, norm=args.norm), temporary)
    temporary.replace(path)


def run_epoch(model, loader, criterion, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    meter = DetectionMetrics(args.match_iou)
    sums = {name: 0.0 for name in ('total', 'person', 'background', 'box', 'auxiliary')}
    seen = 0
    with torch.set_grad_enabled(training):
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(device)
            targets = move_targets(targets, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            losses = criterion(predictions, targets, args.image_size)
            if not all(torch.isfinite(losses[name]).item() for name in sums):
                raise RuntimeError('손실에 NaN/Inf가 발생했습니다.')
            if training:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            else:
                detections = postprocess(predictions, args.image_size, args.score, args.nms_iou)
                meter.update(detections, targets)
            count = len(images)
            seen += count
            for name in sums:
                sums[name] += float(losses[name].detach())*count
            if step == 1 or step % 50 == 0 or step == len(loader):
                label = 'train' if training else 'val'
                print(f'  {label} {step}/{len(loader)} loss={float(losses["total"].detach()):.4f}', flush=True)
    return {name: value/seen for name, value in sums.items()}, meter.result()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=root/'dataset/new_images')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=800)
    parser.add_argument('--lr', type=float, default=3e-4)  #1e-4
    parser.add_argument('--weight-decay', type=float, default=0.005)  # 초기 1e-4, AdamW가 가중치 크기를 억제하는 정도(높이면, 규제가 강해짐, 학습 방해 가능성 있음)
    parser.add_argument('--background-weight', type=float, default=1.15)  # 배경 손실, 배경을 사람으로 잘못 판단한 오차를 전체 손실에 반영하는 값(높이면 배경 오검출을 줄이는 쪽, 사람의 점수도 낮아질 수 있음)
    parser.add_argument('--box-weight', type=float, default=3.0)  # 초기 5.0, 전체 손실에서 박스 위치,크기 오차에 주는 가중치, (높이면 박스를 정확하게 맞추는 오차를 더 중요하게 반영)
    parser.add_argument('--score', type=float, default=0.59)  # 초기 0.5, 사람 후보를 남길 최소 예측 점수, (높이면, 박스가 줄지만, 사람을 더 놓칠 수 있음)
    parser.add_argument('--nms-iou', type=float, default=0.44)  # 초기 0.5, 예측끼리 중복 박스를 제거하는 겹침 기준, (높이면, 더 많이 겹쳐야 제거하므로, 박스가 더 많이 남을 수 있음)
    parser.add_argument('--match-iou', type=float, default=0.50)  # 초기 0.5, 예측과 정답이 일치했다고 인정할 겹침 기준, (높이면 박스 위치가 더 정확해야 정답 인정)
    parser.add_argument('--patience', type=int, default=50, help='0이면 조기 종료 해제')
    parser.add_argument('--monitor', choices=('f1', 'val_loss'), default='f1')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--augment', action='store_true', help='학습에만 수평 뒤집기/밝기/대비 적용')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output', type=Path)
    # 추가된 옵션
    parser.add_argument('--norm', choices=('bn', 'gn'), default='gn')
    parser.add_argument('--box-loss', choices=('giou', 'ciou'), default='ciou')
    parser.add_argument('--obj-loss', choices=('bce', 'focal'), default='bce')
    # 학습률 스케줄러 추가(scheduler / 2026.11.24)
    parser.add_argument('--scheduler', choices=('none', 'cosine'), default='cosine')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.limit < 0 or args.patience < 0:
        parser.error('epochs/batch-size는 양수, limit/patience는 0 이상이어야 합니다.')
    if args.image_size < 64 or args.image_size % 32:
        parser.error('image-size는 64 이상인 32의 배수여야 합니다.')
    if not all(math.isfinite(v) and v > 0 for v in (args.lr, args.box_weight, args.background_weight)):
        parser.error('lr와 손실 가중치는 유한한 양수여야 합니다.')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('weight-decay는 유한한 0 이상 값이어야 합니다.')
    if not all(0 <= v <= 1 for v in (args.score, args.nms_iou)) or not 0 < args.match_iou <= 1:
        parser.error('score/nms-iou는 0~1, match-iou는 0 초과 1 이하여야 합니다.')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    samples = load_dataset(args.dataset)
    groups = split_dataset(samples)
    if args.limit:
        groups = tuple(group[:args.limit] for group in groups)
    train, val, test = groups
    if not train or not val:
        raise ValueError('학습/검증 그룹이 비어 있습니다.')
    output = args.output or root/'checkpoints_multiscale'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'dataset': str(args.dataset.resolve()),
              'output': str(output.resolve()), 'format': FORMAT, 'device': str(device)}
    (output/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {name: [str(s['image'].resolve()) for s in group]
                for name, group in zip(('train', 'val', 'test'), groups)}
    (output/'split.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Device: {device} | train={len(train)} val={len(val)} test={len(test)}', flush=True)
    print(f'Output: {output.resolve()}', flush=True)
    train_loader = DataLoader(PersonDataset(train, args.image_size, args.augment),
                              batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_samples)
    val_loader = DataLoader(PersonDataset(val, args.image_size), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_samples)
    # 수정된 부분(norm, box_loss, obj_loss 추가 / 26.11.23)
    model = create_person_detection(norm=args.norm).to(device)
    criterion = DetectionLoss(
        args.box_weight, 
        args.background_weight,
        box_loss = args.box_loss,
        obj_loss=args.obj_loss
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, eps=1e-7)
    # 학습률 스케줄 추가 (26.11.24)
    scheduler = None
    if args.scheduler == 'cosine':
        warmup_epochs = min(5, max(1, args.epochs // 10))
        if args.epochs > warmup_epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs), eta_min=1e-6
            )
    
    best_loss, best_f1, stale = float('inf'), -1.0, 0
    with (output/'history.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(('epoch', 'loss', 'val_loss', 'val_person', 'val_background',
                         'val_box', 'val_auxiliary', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'))
        for epoch in range(1, args.epochs+1):
            print(f'\nEpoch {epoch}/{args.epochs}', flush=True)
            training, _ = run_epoch(model, train_loader, criterion, device, args, optimizer)
            # 학습 스케쥴러 추가(26.11.24)
            if scheduler is not None:
                scheduler.step()

            validation, metrics = run_epoch(model, val_loader, criterion, device, args)
            writer.writerow((epoch, training['total'], validation['total'],
                             validation['person'], validation['background'], validation['box'],
                             validation['auxiliary'], *(metrics[k] for k in
                             ('tp', 'fp', 'fn', 'precision', 'recall', 'f1'))))
            file.flush()
            print(f"loss={training['total']:.5f} val_loss={validation['total']:.5f} "
                  f"person={validation['person']:.4f} background={validation['background']:.4f} "
                  f"box={validation['box']:.4f}", flush=True)
            print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}", flush=True)
            improved_loss = validation['total'] < best_loss
            improved_f1 = metrics['f1'] > best_f1
            save_checkpoint(output/'last.pt', model, optimizer, epoch, validation, metrics, args)
            if improved_loss:
                best_loss = validation['total']
                save_checkpoint(output/'best.pt', model, optimizer, epoch, validation, metrics, args)
            if improved_f1:
                best_f1 = metrics['f1']
                save_checkpoint(output/'best_f1.pt', model, optimizer, epoch, validation, metrics, args)
                print('best_f1.pt saved', flush=True)
            improved = improved_f1 if args.monitor == 'f1' else improved_loss
            stale = 0 if improved else stale+1
            if args.patience and stale >= args.patience:
                print(f'Early stopping: {args.monitor}', flush=True)
                break
    print(f'Completed: {output.resolve()/"best_f1.pt"}', flush=True)


if __name__ == '__main__':
    main()
