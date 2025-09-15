import cv2
import numpy as np
import threading
from app import config
from app.models.arrow_detector import ArrowDetector
from app.models.target_detector import TargetDetector

arrow_model = ArrowDetector()
target_model = TargetDetector()

# 과녁 코너 좌표 캐싱
target_corners = None
corner_lock = threading.Lock()


def generate_frames():
    global target_corners
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
                # ======================
                # 1) 과녁 탐지 (최초 1회만)
                # ======================
                if target_corners is None:   # 아직 탐지 안 됐을 때만 실행
                    t_result = target_model.predict(frame)
                    if t_result.masks is not None:
                        for m in t_result.masks.data:
                            mask = m.cpu().numpy().astype(np.uint8) * 255
                            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                            contours, _ = cv2.findContours(
                                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )

                            for cnt in contours:
                                epsilon = 0.01 * cv2.arcLength(cnt, True)
                                approx = cv2.approxPolyDP(cnt, epsilon, True)

                                # 코너 좌표 캐싱
                                with corner_lock:
                                    target_corners = approx

                                # 한번만 탐지했으면 break
                                break

                # 캐싱된 좌표가 있으면 항상 그려주기
                with corner_lock:
                    if target_corners is not None:
                        cv2.polylines(frame, [target_corners], True, (0, 0, 255), 2)

                # --------------------
                # 2) 화살 탐지
                # --------------------
                a_results = arrow_model.predict(frame)
                for box in a_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    pad = 10
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = x2 + pad, y2 + pad

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

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
