import tensorflow as tf
from tensorflow.keras import layers


def create_person_detection():
    inputs = tf.keras.Input(shape=(256, 256, 3))

    # 이미지 특징 추출 cnn
    x = layers.Conv2D(32, 3, activation='relu')(inputs)  # 커널수, 커널 크기
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # 드롭아웃
    x = layers.Dropout(0.2)(x)  # 학습 중 이 층의 출력값 20%를 무작위로 끔

    # 위치마다 5개(사람 존재 점수, 중심 x, 중심 y, 너비, 높이)의 값 출력:
    x = layers.Conv2D(
        filters=25,  # 후보 5개 × 후보별 값 5개
        kernel_size=1,
        activation="sigmoid",
        name="person_predictions",
    )(x)

    outputs = layers.Reshape((14, 14, 5, 5))(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

