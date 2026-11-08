import cv2

# 0은 일반적으로 노트북의 기본 카메라입니다.
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

        # 읽어온 이미지를 창에 표시합니다.
        cv2.imshow("Camera Test", flipped_frame)

        # 영상 창을 선택한 뒤 q를 누르면 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()