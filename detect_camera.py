import cv2
from ultralytics import YOLO

# 처음 실행할 때 사전 학습된 가중치가 다운로드될 수 있습니다.
model = YOLO("yolo11n.pt")

# 0은 일반적으로 기본 웹캠입니다.
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다. 연결과 권한을 확인하세요.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("카메라에서 프레임을 읽지 못했습니다.")
            break

        # 각 프레임에서 객체를 탐지합니다.
        results = model(frame, conf=0.5, verbose=False)

        # 탐지 결과를 원본 영상 위에 표시합니다.
        annotated_frame = results[0].plot()

        cv2.imshow("Object Detection", annotated_frame)

        # q 키를 누르면 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()