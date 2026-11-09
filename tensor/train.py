from pathlib import Path

import tensorflow as tf

from model.losses import detection_loss
from model.network import create_person_detection
from prepare_dataset import load_dataset, prepare_samples, split_dataset


if __name__ == "__main__":
    weights_path = Path(__file__).resolve().parent / "person_detector_v3.weights.h5"
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=detection_loss,
    )

    # 학습 중단(조기 종료) 콜백 설정
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=30,  # 15번 연속(에포크)동안 학습률 개선이 없으면 학습 중단
    restore_best_weights=True,
    verbose=1,  # 학습 중단 시점의 가중치를 복원, 조기 종료 메시지 출력(기본값-0: 출력안함)
)

    # 학습 실행
    model.fit(
        train_images,
        train_targets,
        validation_data=(val_images, val_targets),
        epochs=80,
        batch_size=10,
        callbacks=[early_stopping],
    )

    # 학습된 가중치 저장
    model.save_weights(str(weights_path))

    print(f"학습 완료: {weights_path}")
