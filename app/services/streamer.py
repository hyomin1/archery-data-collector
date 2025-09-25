import cv2
import numpy as np
import threading
from app.core import config
from app.models.arrow_detector import ArrowDetector
from app.models.target_detector import TargetDetector

arrow_model = ArrowDetector()
target_model = TargetDetector()

# 카메라별 과녁 좌표 캐싱
target_corners = {}
corner_locks = {}


def generate_frames(stream_url, cam_id=0):
    global target_corners, corner_locks

    if cam_id not in target_corners:
        target_corners[cam_id] = None
        corner_locks[cam_id] = threading.Lock()

    cap = None
    try:
        cap = cv2.VideoCapture(stream_url,cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[카메라 {cam_id}] RTSP 연결 실패 (모니터링 불가)")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # ======================
                # 1) 과녁 탐지 (최초 1회만)
                # ======================
                if target_corners[cam_id] is None:
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

                                with corner_locks[cam_id]:
                                    target_corners[cam_id] = approx
                                break

                # 항상 코너 그리기
                with corner_locks[cam_id]:
                    if target_corners[cam_id] is not None:
                        cv2.polylines(frame, [target_corners[cam_id]], True, (0, 0, 255), 2)

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
                print(f"[카메라 {cam_id}] 추론 에러:", e)

            # 스트리밍 전송
            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    finally:
        if cap:
            cap.release()
        print(f"[카메라 {cam_id}] 모니터링 세션 종료")
