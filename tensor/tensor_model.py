import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers

from prepare_dataset import load_dataset, split_dataset

# 이미지 읽어오는 함수
def prepare_samples(samples):
    images = []
    boxes = []

    for sample in samples:
        # 이미지 읽기, image와 xml 중 이미지 만을 가지고 옴
        image = cv2.imread(str(sample["image"]))

        # 이미지 로드여부 확인
        if image is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {sample['image']}")

        # 학습에 사용할 이미지 형태로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = image.astype("float32") / 255.0

        # 14×14 그리드의 각 칸에 후보 5개와 후보별 값 5개를 저장
        target_boxes = np.zeros((14, 14, 5, 5), dtype=np.float32)

        for box in sample["boxes"]:
            x_min, y_min, x_max, y_max = box

            # 그리드 좌표 계산
            # 왼쪽 위, 오른쪽 아래 좌표를 중심/크기로 변환
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # 사람 중심이 속한 14x14 위치
            column = min(int(center_x * 14), 13)
            row = min(int(center_y * 14), 13)

            # 같은 위치의 빈 후보 칸을 찾기
            for slot in range(5):
                if target_boxes[row, column, slot, 0] == 0:  # 존재 점수가 0이면 빈 칸
                    target_boxes[row, column, slot] = [
                        1.0,  # 존재 점수
                        center_x * 14 - column,  # 그리드 내 상대 좌표
                        center_y * 14 - row,  # 그리드 내 상대 좌표
                        width,
                        height,
                    ] 
                    break
            else:
                raise ValueError(
                        f"한 위치에 사람이 5명보다 많습니다: {sample['image']}"
                    )

        images.append(image)
        boxes.append(target_boxes)

    return np.array(images), np.array(boxes)


# 사람 탐지 모델 생성
def create_person_detection():
    inputs = tf.keras.Input(shape=(256, 256, 3))

    # 이미지 특징 추출 cnn
    x = layers.Conv2D(32, 3, activation='relu')(inputs)  # 커널수, 커널 크기
    x = layers.MaxPooling2D(pool_size=2)(x)  # 256 -> 254 -> 127

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  # 127 -> 125 -> 62

    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  # 62 -> 60 -> 30

    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  # 30 -> 28 -> 14

    # 위치마다 5개(사람 존재 점수, 중심 x, 중심 y, 너비, 높이)의 값 출력:
    x = layers.Conv2D(
        filters=25,  # 후보 5개 × 후보별 값 5개
        kernel_size=1,
        activation="sigmoid",
        name="person_predictions",
    )(x)

    outputs = layers.Reshape((14, 14, 5, 5))(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

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

if __name__ == "__main__":
    weights_path = Path(__file__).resolve().parent / "person_detector_v2.weights.h5"
    if weights_path.exists():
        raise FileExistsError(f"기존 가중치를 보존하기 위해 학습을 중단했습니다: {weights_path}")
    # 모델 생성
    model = create_person_detection()
    model.summary()

    # 이미지와 XML 읽기
    samples = load_dataset()

    # 기존 코드에서 만든 무작위 분리 사용
    train_samples, val_samples, test_samples = split_dataset(samples)

    # 이미지와 정답을 모델 입력 형태로 변환
    train_images, train_targets = prepare_samples(train_samples)
    val_images, val_targets = prepare_samples(val_samples)
    test_images, test_targets = prepare_samples(test_samples)

    print("학습 이미지:", train_images.shape)
    print("학습 정답:", train_targets.shape)

    print("검증 이미지:", val_images.shape)
    print("검증 정답:", val_targets.shape)

    print("테스트 이미지:", test_images.shape)
    print("테스트 정답:", test_targets.shape)

    # 학습 방법 설정
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=detection_loss,
    )

    # 학습 실행
    model.fit(
        train_images,
        train_targets,
        validation_data=(val_images, val_targets),
        epochs=80,
        batch_size=4,
    )

    # 학습된 가중치 저장
    model.save_weights(str(weights_path))

    print(f"학습 완료: {weights_path}")
