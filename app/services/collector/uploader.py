import os, cv2, threading, datetime, requests
from queue import Queue
from app.core.logger import get_logger
from app.core import config

logger = get_logger(__name__)
upload_queue = Queue()

API_URL = f"https://api.roboflow.com/dataset/{config.ROBOFLOW_PROJECT}/upload"
API_KEY = config.ROBOFLOW_API_KEY

def draw_boxes(frame, boxes):
    """YOLO 박스를 프레임 위에 그림"""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def upload_worker():
    while True:
        task = upload_queue.get()
        if task is None:
            break

        cam_id, event_time, event_frames = task
        timestamp = event_time.strftime("%Y%m%d_%H%M%S")

        for idx, (f, det, boxes) in enumerate(event_frames):
            status = "yolo_detected" if det else "undetected"
            filename = f"{cam_id}_{timestamp}_f{idx}_{status}.jpg"

            # === 감지된 경우 bbox를 그려줌 ===
            if det and boxes is not None:
                f = draw_boxes(f, boxes)

            success, buffer = cv2.imencode(".jpg", f)
            if not success:
                logger.error(f"[{cam_id}] 인코딩 실패: {filename}")
                continue

            files = {"file": (filename, buffer.tobytes(), "image/jpeg")}
            params = {"api_key": API_KEY}

            try:
                resp = requests.post(API_URL, files=files, params=params)
                resp.raise_for_status()
                logger.info(f"[{cam_id}] 업로드 성공: {filename}")
            except Exception as e:
                logger.error(f"[{cam_id}] 업로드 실패: {filename}, {e}")

        upload_queue.task_done()

num_workers = len(config.CAMERA_URLS)  
for i in range(num_workers):
    threading.Thread(target=upload_worker, daemon=True).start()
    logger.info(f"업로드 워커 {i+1} 시작됨")


def upload_event(cam_id, event_time, event_frames):
    """
    event_frames: [(frame, detected, boxes), ...]
    """
    upload_queue.put((cam_id, event_time, event_frames))
