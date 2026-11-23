import torch
import torch.nn.functional as F


def detection_loss(y_true, y_pred):
    # 두 입력의 형태: (배치, 30, 30, 후보 5개, 값 5개).
    # 마지막 축의 순서: [사람 존재 여부, 칸 안 중심 x, 칸 안 중심 y, 너비, 높이].
    # 호출 순서는 detection_loss(정답, 예측)이며 반환값은 배치 전체의 손실 스칼라다.
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"정답/예측 형태 불일치: {y_true.shape}, {y_pred.shape}"
        )

    # 정답을 예측과 같은 CPU/GPU 및 자료형으로 맞춰 함께 계산할 수 있게 한다.
    y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)

    # ...는 앞의 배치/행/열/후보 축을 유지한다는 뜻이다.
    # 정답 존재 여부는 0 또는 1, 예측 존재 점수는 sigmoid가 출력한 확률이다.
    true_objectness = y_true[..., 0]
    pred_objectness = y_pred[..., 0]

    # sigmoid 결과가 0 또는 1에 가까울 때 수치 안정성 확보
    eps = 1e-7
    pred_objectness = pred_objectness.clamp(eps, 1.0 - eps)

    # 각 후보가 사람인지 배경인지에 대한 이진 교차 엔트로피.
    # reduction="none"으로 평균을 내지 않아 사람과 배경을 따로 집계한다.
    objectness_error = F.binary_cross_entropy(
        pred_objectness,
        true_objectness,
        reduction="none",
    )

    # 마스크를 곱하면 해당 종류의 후보에 대한 오차만 남는다.
    person_mask = true_objectness
    background_mask = 1.0 - true_objectness

    # 배치 전체의 사람/배경 후보 수. 최소 분모를 1로 두어 0으로 나누지 않는다.
    # 해당 종류의 후보가 없으면 마스크가 모두 0이라 분자와 손실도 0이다.
    person_count = person_mask.sum().clamp_min(1.0)
    background_count = background_mask.sum().clamp_min(1.0)

    # 사람 후보의 존재 점수 오차 평균.
    person_loss = (
        objectness_error * person_mask
    ).sum() / person_count

    # 배경 후보의 존재 점수 오차 평균.
    # 두 종류를 따로 평균내므로 많은 배경 후보가 사람 오차를 희석하지 않는다.
    background_loss = (
        objectness_error * background_mask
    ).sum() / background_count

    # 존재 여부를 제외한 좌표 4개만 꺼낸다.
    true_boxes = y_true[..., 1:5]
    pred_boxes = y_pred[..., 1:5]

    # 각 후보의 좌표 4개에 대한 제곱 오차를 합산한다.
    box_error = torch.square(true_boxes - pred_boxes).sum(dim=-1)
    # 배경 후보에는 실제 박스가 없으므로 좌표 오차를 계산에 포함하지 않는다.
    box_loss = (box_error * person_mask).sum() / person_count

    # 세 항을 같은 가중치로 합산한다. 학습 루프에서 loss.backward()를 호출하면
    # 예측값을 만든 모델 파라미터까지 기울기가 역전파된다.
    return person_loss + background_loss + box_loss
