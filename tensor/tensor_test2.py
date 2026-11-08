import tensorflow as tf
from tensorflow.keras import layers

# 사람 탐지 모델 생성
def create_person_detection():
    inputs = tf.keras.Input(shape=(256, 256, 3))

    # 이미지 특징 추출 cnn
    x = layers.Conv2D(32, 3, activation='relu')(inputs)  # 커널수, 커널 크기
    x = layers.MaxPooling2D(pool_size=2)(x)  # 256 -> 128

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  # 128 -> 64

    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  # 64 -> 32

    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  # 32 -> 16

    # 위치마다 5개(사람 존재 점수, 중심 x, 중심 y, 너비, 높이)의 값 출력:
    outputs = layers.Conv2D(
        filters=5,
        kernel_size=1,
        activation="sigmoid",
        name="person_predictions",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = create_person_detection()

# 모델 구조와 입력·출력 크기 확인
model.summary()

# 모델에 가상의 이미지 한 장을 넣어 출력 형태 확인
example_image = tf.zeros((1, 256, 256, 3))
predictions = model(example_image, training=False)

print("입력 형태:", example_image.shape)
print("출력 형태:", predictions.shape)