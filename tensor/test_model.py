import cv2

from detect_webcam import WEIGHTS, SCORE_THRESHOLD, decode_predictions
from prepare_dataset import load_dataset, split_dataset
from tensor_model import create_person_detection


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {WEIGHTS}")

    _, _, test_samples = split_dataset(load_dataset())
    model = create_person_detection()
    model.load_weights(str(WEIGHTS))
    print(f"테스트 이미지 {len(test_samples)}장 | 초록: 정답, 빨강: 예측 | d: 다음, a: 이전, q: 종료")

    index = 0
    try:
        while True:
            sample = test_samples[index]
            image = cv2.imread(str(sample["image"]))
            if image is None:
                raise RuntimeError(f"이미지를 읽을 수 없습니다: {sample['image']}")

            height, width = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(rgb, (256, 256)).astype("float32") / 255.0
            predictions = model(input_image[None, ...], training=False).numpy()[0]
            boxes, scores = decode_predictions(predictions, SCORE_THRESHOLD)

            for x_min, y_min, x_max, y_max in sample["boxes"]:
                cv2.rectangle(image, (int(x_min * width), int(y_min * height)),
                              (int(x_max * width), int(y_max * height)), (0, 255, 0), 2)
            for (y_min, x_min, y_max, x_max), score in zip(boxes, scores):
                left, top = int(x_min * width), int(y_min * height)
                cv2.rectangle(image, (left, top), (int(x_max * width), int(y_max * height)),
                              (0, 0, 255), 2)
                cv2.putText(image, f"{score:.2f}", (left, max(top - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            print(f"[{index + 1}/{len(test_samples)}] {sample['image'].name} | 정답 {len(sample['boxes'])}, 예측 {len(boxes)}")
            cv2.imshow("Test: green = truth, red = prediction", image)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d") and index < len(test_samples) - 1:
                index += 1
            elif key == ord("a") and index > 0:
                index -= 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
