import cv2
import tensorflow as tf

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "카메라를 열 수 없습니다. 카메라 연결과 접근 권한을 확인하세요."
    )

try:
    while True:
        # 카메라에서 이미지 한 장을 읽습니다.
        success, frame = camera.read()

        if not success:
            print("카메라에서 영상을 읽지 못했습니다.")
            break

        # 카메라 좌우 반전
        flipped_frame = cv2.flip(frame, 1)

        # 전처리
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (1024, 1024))
        normalized_frame = resized_frame.astype("float32") / 255.0

        # 텐서변환과 이미지 개수 차원 변화
        input_tensor = tf.convert_to_tensor(normalized_frame[None, ...])

        # 읽어온 이미지를 창에 표시합니다.
        cv2.imshow("Camera Test", flipped_frame)

        key = cv2.waitKey(1) & 0xFF

        # s를 누르면 변환된 텐서의 정보 출력
        if key == ord("s"):
            print("텐서 형태:", input_tensor.shape)
            print("데이터 종류:", input_tensor.dtype)

        # 영상 창을 선택한 뒤 q를 누르면 종료합니다.
        if key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()