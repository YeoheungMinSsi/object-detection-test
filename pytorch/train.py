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
from model.losses import detection_loss, hard_negative_loss
from metrics import DetectionMetrics
from prepare_dataset import load_dataset, split_dataset, PersonDataset


def save_checkpoint(path, model, optimizer, epoch, val_loss, metrics=None):
    # 임시 파일 저장이 완료된 뒤 교체해 이전 체크포인트를 보호한다.
    temporary = path.with_suffix('.tmp')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'metrics': metrics,
        'input_shape': [3, 512, 512],
        'output_shape': [30, 30, 5, 5],
    }, temporary)
    temporary.replace(path)


def run_epoch(model, loader, device, optimizer=None, loss_fn=detection_loss,
              metrics=None):
    training = optimizer is not None
    model.train(training)  # 검증할 때는 dropout을 끈다.
    total_loss = 0.0
    sample_count = 0
    with torch.set_grad_enabled(training):
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = loss_fn(targets, predictions)
            if metrics is not None:
                metrics.update(predictions.detach(), targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError('손실이 NaN 또는 Inf입니다. 학습을 중단합니다.')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                               error_if_nonfinite=True)
                optimizer.step()
            total_loss += loss.item() * images.shape[0]
            sample_count += images.shape[0]
            if step == 1 or step % 50 == 0 or step == len(loader):
                label = '학습' if training else '검증'
                print(f'  {label} {step}/{len(loader)} loss={loss.item():.4f}', flush=True)
    # 로그는 배치 손실의 이미지 수 가중 평균이다.
    return total_loss / sample_count


def main():
    # NOTE: 학습 설정 값 읽기
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # 에포크
    parser.add_argument('--batch-size', type=int, default=10)  # 배치 사이즈 (전체 데이터 / n)
    parser.add_argument('--lr', type=float, default=0.0001)  # 학습률
    parser.add_argument('--limit', type=int, default=0)  # 학습/검증/테스트 각 그룹에서 사용할 이미지수의 상한 (0: 전체 사용)
    parser.add_argument('--cpu', action='store_true')  # --cpu 사용 확인(실행시 --cpu를 입력하면 cpu사용 아니면 gpu사용)
    parser.add_argument('--patience', type=int, default=200)  # 조기 종료 기준(n에포크 동안 학습이 진행되지 않으면 조기 종료)
    parser.add_argument('--output', type=Path)  # 가중치, 실행 기록을 저장할 폴더
    parser.add_argument('--dataset', type=Path, help='images 및 Annotations가 있는 VOC 폴더')  # 이미지와 라벨이 학습데이터 위치 폴더
    parser.add_argument('--loss', choices=('original', 'hard-negative'), default='hard-negative')  # 손실방식 / original, hard-negative
    parser.add_argument('--score', type=float, default=0.5, help='검증 점수 기준')  # 검증에서 예측 박스를 남길 사람 점수 기준(0.5점 넘기면 박스 표시)
    # IoU: 겹친 면적 / 두 박스가 차지한 전체 합집합 면적
    parser.add_argument('--nms-iou', type=float, default=0.5)  # 중복 예측 박스를 제거할 겹침 기준
    parser.add_argument('--match-iou', type=float, default=0.5)  # 예측 박스를 정답과 일치했다고 인정할 겹침 기준
    parser.add_argument('--monitor', choices=('val_loss', 'f1'), default='val_loss',  # 조기 종료 여부를 판단할 검증 지표
                        help='조기 종료 기준. F1은 클수록 좋음')
    args = parser.parse_args()

    # NOTE: 학습 설정 값이 제대로 입력되었는지 검토하는 부분
    if (args.epochs < 1 or args.batch_size < 1 or args.limit < 0
            or args.patience < 1 or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error('epochs/batch-size/patience/lr는 양수, limit는 0 이상이어야 합니다.')
    if not all(0 <= value <= 1 for value in (args.score, args.nms_iou)) or not 0 < args.match_iou <= 1:
        parser.error('score/nms-iou는 0~1, match-iou는 0 초과 1 이하여야 합니다.')

    # NOTE: 모델 초기화(가중치 초기화)와 데이터 순서 난수(데이터를 어떤 순서로 학습할지 섞음/에포크마다 섞음)등의 
    # 난수를 고정해 실험을 비교하기 쉽게 함, 정수 값은 랜덤하게 적용해도 됨
    random.seed(42)  # 난수의 출발점 seed는 고정
    np.random.seed(42)
    torch.manual_seed(42)

    # NOTE: 학습을 사용할 장치가 cpu인지 gpu인지 확인, --cpu를 지정하면 cpu를 사용
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('사용 장치:', device, flush=True)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), flush=True)

    # NOTE: 학습에 사용할 데이터 로딩
    print('이미지 경로와 라벨 읽는 중...', flush=True)
    groups = split_dataset(load_dataset(args.dataset))
    if args.limit:
        groups = tuple(group[:args.limit] for group in groups)
    # 학습, 검증, 테스트 데이터로 분류 및 데이터 검토
    train_samples, val_samples, test_samples = groups
    if not train_samples or not val_samples:
        raise ValueError('학습과 검증 데이터가 각각 1장 이상 필요합니다.')
    print(f'학습 {len(train_samples)}, 검증 {len(val_samples)}, 테스트 {len(test_samples)}장', flush=True)

    # NOTE: 실행 폴더와 설정 기록 / 가중치를 저장할 새 폴더 정함
    output = args.output or (Path(__file__).resolve().parent / 'checkpoints' /
                            datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    # 실행마다 새 폴더를 사용해 이전 가중치를 보존
    output.mkdir(parents=True, exist_ok=False)  # 이미 같은 폴더가 있으면 덮어쓰기 막음
    manifest = {name: [str(s['image']) for s in group]
                for name, group in zip(('train', 'val', 'test'), groups)}
    # 두 파일(split.json/어떤 이미를 어떤 그룹에 사용했는지,  config.json/학습률,배치 등 실행 설정) 저장
    (output / 'split.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                     encoding='utf-8')
    (output / 'config.json').write_text(json.dumps(
        {**vars(args), 'dataset': str(args.dataset.resolve()) if args.dataset else None,
         'output': str(output), 'device': str(device)},
        ensure_ascii=False, indent=2), encoding='utf-8')

    # NOTE: Windows에서는 먼저 num_workers=0으로 안정적으로 실행, suffle(true)을 통해서 학습 데이터 순서 섞음
    # PersonDataset: 이미지 한 장과 정답을 읽고 변환 / DataLoader: 여러 장을 배치로 묶어서 전달
    train_loader = DataLoader(PersonDataset(train_samples), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(PersonDataset(val_samples), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    model = create_person_detection().to(device)  # network에 있는 구조를 통해서 모델 생성

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,  # learning_rate: 가중치를 한 번에 얼마나 바꿀지 정하는 학습률
        weight_decay=1e-4,  # 가중치가 지나치게 커지지 않도록 줄이는 정규화 강도 -> 과적합 방지
        eps=1e-7,  # 계산시 분모가 너무 작아지는 것을 막음 -> 기울기의 크기
    )

    # NOTE: 에포크를 반복하며 학습,검증하고 결과를 기록한 뒤 가장 좋은 모델(가중치)를 저장하는 부분
    best_loss = float('inf')  # 가장 낮은 검증 손실 / 무한대에서 시작해서 첫 정상 검증값이 기록됨
    best_f1 = -1.0  # 지금까지 가장 높은 검증 F1 값, F1은 0~1이므로 -1으로 시작하면 첫 결과를 저장 가능
    training_loss_fn = hard_negative_loss if args.loss == 'hard-negative' else detection_loss  # 학습에 사용될 함수 선택
    stale_epochs = 0  # 선택한 검증 지표가 개선되지 않은 연속 에포크 수 / 조기 종료에 사용되는 에포크
    
    print('가중치 저장 폴더:', output, flush=True)  # 저장 위치 출력
    
    with (output / 'history.csv').open('w', newline='', encoding='utf-8') as file:  # csv를 작성하도록 설정
        writer = csv.writer(file)
        writer.writerow(('epoch', 'loss', 'val_loss', 'tp', 'fp', 'fn',
                         'precision', 'recall', 'f1'))
        file.flush()

        # NOTE: 에포크 반복/학습 부분
        for epoch in range(1, args.epochs + 1):  # 앞서 add_argument에서 설정한 값이 epochs 객체에 설정 값이 저장됨
            print(f'\nEpoch {epoch}/{args.epochs}', flush=True)

            # run_epoch: 학습 데이터 한번에 처리(예측 -> 손실 계산 -> 역전파 -> 가중치 수정)을 한 배치마다 반복
            # train_loss: 해당 에포크의 배치 손실을 이미지 수로 가중 평균한 값
            train_loss = run_epoch(model, train_loader, device, optimizer,
                                   loss_fn=training_loss_fn)
            # 에포크 전용 지표 집계 객체
            meter = DetectionMetrics(args.score, args.nms_iou, args.match_iou)
            
            # 실험 간 비교를 위해 val_loss는 항상 기존 손실 함수로 계산
            # optimizer를 전달하지 않기 때문에, 평가 모드로 실행하며 가중치 수정은 하지 않음
            val_loss = run_epoch(model, val_loader, device, metrics=meter)
            metrics = meter.result()

            # 에포크 결과 기록 및 출력
            writer.writerow((epoch, train_loss, val_loss, metrics['tp'], metrics['fp'],
                             metrics['fn'], metrics['precision'], metrics['recall'], metrics['f1']))
            file.flush()
            print(f'loss: {train_loss:.6f} - val_loss: {val_loss:.6f}', flush=True)
            print(f"검증 TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}",
                  flush=True)

            # 성능 개선과 상관없이 완료한 에포크마다 last.pt를 갱신
            save_checkpoint(output / 'last.pt', model, optimizer, epoch, val_loss, metrics)
            loss_improved = val_loss < best_loss  # 최솟값과 현재 손실값을 비교, 비교 결과로 true or false를 얻음
            f1_improved = metrics['f1'] > best_f1  # f1 최댓값과 현재 최대 값을 비교, 비교 결과로 true or false를 얻음
            
            # loss_improved에서 현재 얻은 최솟값이 이전 최솟값보다 작으면 기록을 갱신하고 저장
            if loss_improved:
                best_loss = val_loss
                save_checkpoint(output / 'best.pt', model, optimizer, epoch, val_loss, metrics)
                print('best.pt 저장 완료', flush=True)

            # f1_improved에서 현재 얻은 f1이 이전 최댓값보다 크면 기록을 갱신하고 저장
            if f1_improved:
                best_f1 = metrics['f1']
                save_checkpoint(output / 'best_f1.pt', model, optimizer, epoch, val_loss, metrics)
                print('best_f1.pt 저장 완료', flush=True)
            
            # monitor로 선택한 지표의 개선 여부를 사용(조기 종료 기준이 적용)
            improved = f1_improved if args.monitor == 'f1' else loss_improved
            
            # 최고 기록을 갱신하면 대기 횟수를 초기화하고, 아니면 1을 증진
            if improved:
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f'검증 {args.monitor} 개선이 없어 조기 종료합니다.', flush=True)
                break
    print('학습 완료. 검출 F1 기준:', output / 'best_f1.pt', flush=True)
    print('검증 손실 기준:', output / 'best.pt', flush=True)


if __name__ == '__main__':
    main()
