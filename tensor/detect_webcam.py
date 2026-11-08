from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from tensor_model import create_person_detection


WEIGHTS = Path(__file__).resolve().parent / "person_detector_v2.weights.h5"
SCORE_THRESHOLD = 0.15


def decode_predictions(predictions, threshold=SCORE_THRESHOLD):
    # 모델은 각 칸의 상대 중심 좌표와 전체 이미지 기준 너비·높이를 출력한다.
    rows, columns, slots = np.where(predictions[..., 0] >= threshold)
    values = predictions[rows, columns, slots]
    center_x = (columns + values[:, 1]) / predictions.shape[1]
    center_y = (rows + values[:, 2]) / predictions.shape[0]
    boxes = np.stack([
        center_y - values[:, 4] / 2,
        center_x - values[:, 3] / 2,
        center_y + values[:, 4] / 2,
        center_x + values[:, 3] / 2,
    ], axis=1)
    boxes = np.clip(boxes, 0, 1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], values[valid, 0]
    selected = tf.image.non_max_suppression(boxes, scores, max_output_size=100, iou_threshold=0.5).numpy()
    return boxes[selected], scores[selected]


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {WEIGHTS}")
    model = create_person_detection()
    model.load_weights(str(WEIGHTS))
    threshold = SCORE_THRESHOLD

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("카메라를 열 수 없습니다. 카메라 연결과 권한을 확인하세요.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("카메라 영상을 읽지 못했습니다.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(rgb, (256, 256)).astype("float32") / 255.0
            predictions = model(image[None, ...], training=False).numpy()[0]
            boxes, scores = decode_predictions(predictions, threshold)

            height, width = frame.shape[:2]
            for (y_min, x_min, y_max, x_max), score in zip(boxes, scores):
                left, top = int(x_min * width), int(y_min * height)
                right, bottom = int(x_max * width), int(y_max * height)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"person? {score:.2f}", (left, max(top - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            max_score = predictions[..., 0].max()
            cv2.putText(frame, f"max: {max_score:.2f}  threshold: {threshold:.2f} (+/-)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Person detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("+"), ord("=")):
                threshold = min(1.0, round(threshold + 0.05, 2))
            elif key == ord("-"):
                threshold = max(0.0, round(threshold - 0.05, 2))
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
