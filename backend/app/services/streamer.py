import cv2
import torch
import threading
from app import config
from app.models.arrow_detector import ArrowDetector
from app.models.target_detector import TargetDetector
from app.models.corner_regressor import load_corner_regressor, transform

# -------------------------------
# 전역 상태
# -------------------------------
corner_model = load_corner_regressor()
arrow_model = ArrowDetector()
target_model = TargetDetector()

# 과녁 코너 좌표 캐싱
target_corners = None
corner_lock = threading.Lock()


def get_target_corners(frame):
    global target_corners
    with corner_lock:
        if target_corners is None:
            frame_resized = cv2.resize(frame, (128, 128))
            tensor = transform(frame_resized).unsqueeze(0).to("cuda")
            with torch.no_grad():
                preds = corner_model(tensor).cpu().numpy()[0]
            h, w, _ = frame.shape
            target_corners = [
                (int(preds[i] * w), int(preds[i + 1] * h))
                for i in range(0, 8, 2)
            ]
            print("과녁 코너 좌표 고정:", target_corners)
    return target_corners


def generate_frames():
    cap = None
    try:
        cap = cv2.VideoCapture(config.STREAM_URL)
        if not cap.isOpened():
            print("RTMP 연결 실패 (모니터링 불가)")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # --------------------
                # 1) 과녁 탐지 + Corner CNN
                # --------------------
                t_results = target_model.predict(frame)
                for box in t_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # crop → CNN
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_resized = cv2.resize(crop, (128, 128))
                    tensor = transform(crop_resized).unsqueeze(0).to("cuda")

                    with torch.no_grad():
                        preds = corner_model(tensor).cpu().numpy()[0]

                    # bbox 기준 좌표 복원
                    corners = []
                    for i in range(0, 8, 2):
                        cx = int(x1 + preds[i] * (x2 - x1))
                        cy = int(y1 + preds[i + 1] * (y2 - y1))
                        corners.append((cx, cy))

                    # 빨간 코너 점 + 테두리
                    for pt in corners:
                        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                    for j in range(4):
                        cv2.line(frame, corners[j], corners[(j + 1) % 4], (0, 0, 255), 2)

                # --------------------
                # 2) 화살 탐지
                # --------------------
                a_results = arrow_model.predict(frame)
                for box in a_results[0].boxes:
                    # 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
                    # 박스 확장 (예: 10픽셀씩 키우기)
                    pad = 10
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = x2 + pad, y2 + pad

                    # 박스 그리기 (두께 4)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

                    # confidence 표시
                    conf = float(box.conf[0]) if box.conf is not None else 0
                    label = f"{conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


            except Exception as e:
                print("추론 에러:", e)

            # 스트리밍 전송
            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    finally:
        if cap:
            cap.release()
        print("모니터링 세션 종료")
