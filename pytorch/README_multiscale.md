# 다중 해상도 사람 검출 코드

기존 model/network.py, backbone.py, neck.py, head.py를 사용한다.
기존 train.py, losses.py, predict.py, metrics.py, test_model.py는 사용하지 않는다.
prepare_dataset.py의 XML 읽기/load_dataset와 split_dataset만 재사용한다.
기존 파일과 가중치는 덮어쓰지 않는다.

## 시작

가상환경을 활성화하고 pytorch 폴더에서:

```powershell
python train_multiscale.py --limit 2 --epochs 1
python train_multiscale.py
```

기본 데이터: dataset/new_images. 기본 입력 512, batch 1, epochs 80,
AdamW lr 0.0001, 검증 F1 기준 patience 20. --patience 0이면 조기 종료 해제.
증강은 기본 해제. --augment를 붙이면 학습에만 수평 뒤집기와 밝기/대비 적용.
--dataset, --cpu, --image-size, --score, --nms-iou, --match-iou도 지원.

기존 고정 seed 42의 60/25/15 분리를 유지한다. 가까운 촬영 장면 사이의
누출을 방지하는 그룹 분리는 구현하지 않았으므로 실제 평가에서는 별도 점검한다.

## 새 가중치 확인

```powershell
python test_multiscale.py --weights checkpoints_multiscale/실행폴더/best_f1.pt
python test_multiscale.py --weights checkpoints_multiscale/실행폴더/best_f1.pt --no-show
python test_multiscale.py --weights checkpoints_multiscale/실행폴더/best_f1.pt --split test
```

기본 split은 val. 초록=정답, 빨강=예측. D/Space 다음, A 이전, Q/Esc 종료.
창을 닫거나 중간 종료해도 확인한 이미지의 결과는 남는다.
inspection 폴더에 JPG, report.csv, summary.json을 저장한다.
summary의 complete=false면 전체 그룹 평가가 아니라 일부 이미지만 확인한 결과다.
검증 score/NMS/IoU는 학습 체크포인트 설정을 기본으로 복원한다.

## 역할

- multiscale/data.py: 정규화 xyxy 박스 목록과 이미지별 가변 정답 배치
- multiscale/targets.py: 크기별 우선 해상도 + 중심 칸 정적 배정
- multiscale/ops.py: 공유 좌표 복원, 점수 필터, NMS, 일대일 매칭 지표
- multiscale/loss.py: logit BCE, 어려운 배경 선택, GIoU, 중심/크기 보조 오차
- train_multiscale.py: 역전파/검증/에포크별 CSV/원자적 체크포인트 교체
- test_multiscale.py: 체크포인트 로딩/저장된 split 평가/그림 저장

## 출력/좌표 규약

raw: small/medium/large 각각 B,5,H,W. 채널 순서 obj_logit,tx,ty,tw,th.
cx=(sigmoid(tx)+column)*stride, cy=(sigmoid(ty)+row)*stride,
w=softplus(tw)*stride, h=softplus(th)*stride.
공유 복원 함수는 좌표를 입력 크기로 나눠 정규화 xyxy를 반환한다.
입력은 정사각형으로 resize하며 원본 종횡비를 보존하는 letterbox는 적용하지 않는다.
이미지와 정답이 동일하게 대응되며, 그림에는 정규화 좌표를 원본 크기로 복원한다.

정답 1개당 양성 위치 1개. 긴 변의 정규화 길이 0.125/0.3125를 경계로
우선 해상도를 선택한다. 충돌 시 다른 해상도의 중심 칸을 시도하고,
모두 점유되면 정답을 버리지 않고 오류를 낸다. 정적 배정의 한계이므로
아주 밀집된 데이터에는 동적 배정 설계가 추가로 필요하다.

총 손실 = 사람 BCE + background_weight*어려운 배경 BCE
          + box_weight*GIoU + 0.5*중심/로그크기 보조 손실.
배경은 이미지별 max(32,3*양성 수)개를 오차순으로 선택한다.
학습과 검증에 같은 손실 함수를 쓰며 CSV의 손실은 배치 손실의 이미지 수 가중 평균이다.

검증 지표는 score/NMS/매칭 IoU를 고정한 precision/recall/F1이며 mAP는 아니다.
NMS 전 상위 1000개, 이후 최대 100개 예측을 사용한다.
best.pt=최저 val_loss, best_f1.pt=최고 F1, last.pt=마지막 완료 에포크.
F1 동점은 갱신하지 않으며 best_f1 저장와 조기 종료는 실제 검출 품질을 보장하지 않는다.

특정 YOLO 버전의 재현이 아닌 사용자 정의 anchor-free 교육용 구성이다.
사전학습 가중치를 사용하지 않는다. 기존 단일 해상도 .pt 파일은 호환되지 않는다.
기본 설정의 품질은 검증 실험으로 판단해야 하며 구조 변경만으로 개선을 보장하지 않는다.
