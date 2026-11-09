import tensorflow as tf

# 각 사람 후보에 대한 오차값의 평균 + 배경 후보 간 오차의 평균 + 네모 박스의 위치 오차 평균을 다 더해서 평균을 구함
# 모든 오차에 대해 더해 평균을 구함 -> 이것을 손실 함수의 오차 값으로 사용
def detection_loss(y_true, y_pred):
    # 정답과 예측에서 사람 존재 여부를 분리
    true_objectness = y_true[..., 0]
    pred_objectness = y_pred[..., 0]

    # 사람이 있는지 없는지에 대한 오차
    objectness_loss = tf.keras.backend.binary_crossentropy(
        true_objectness,
        pred_objectness,
    )

    # 사람이 있는 칸에 더 큰 비중을 둠
    # 비어 있는 칸이 훨씬 많기 때문
    weights = 1.0 + true_objectness * 9.0
    objectness_loss = tf.reduce_mean(objectness_loss * weights)

    # 사람이 있는 칸에서만 사각형 위치 오차 계산
    true_boxes = y_true[..., 1:5]
    pred_boxes = y_pred[..., 1:5]

    box_error = tf.reduce_sum(
        tf.square(true_boxes - pred_boxes),
        axis=-1,
    )

    box_loss = tf.reduce_sum(box_error * true_objectness) / (
        tf.reduce_sum(true_objectness) + 1e-7
    )

    return objectness_loss + box_loss

