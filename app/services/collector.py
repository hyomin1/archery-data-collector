import cv2
import time
import datetime
import threading
import logging
import os
from collections import deque
from queue import Queue
from app import config
from app.models.arrow_detector import ArrowDetector
from logging.handlers import RotatingFileHandler
from ultralytics import YOLO
from roboflow import Roboflow


# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("collector.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

collector_thread = None
stop_event = threading.Event()

# Roboflow 초기화
rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
project = rf.workspace(config.ROBOFLOW_WORKSPACE).project(config.ROBOFLOW_PROJECT)

# === 업로드 큐/워커 ===
upload_queue = Queue()

def upload_worker():
    while True:
        event_frames = upload_queue.get()
        if event_frames is None:  # 종료 신호
            break

        for idx, (f, det) in enumerate(event_frames):
            status = "yolo_detected" if det else "undetected"
            filename = f"frame_{idx}_{status}.jpg"
            filepath = os.path.join(".", filename)

            cv2.imwrite(filepath, f)
            try:
                project.upload(filepath)
                logger.info(f"Roboflow 업로드 완료: {filename}")
            except Exception as e:
                logger.error(f"Roboflow 업로드 실패: {filename}, {e}")
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

        upload_queue.task_done()

# 업로드 전용 스레드 시작
threading.Thread(target=upload_worker, daemon=True).start()


def start_collector():
    global collector_thread, stop_event
    if collector_thread and collector_thread.is_alive():
        logger.warning("이미 스레드가 실행 중입니다.")
        return
    stop_event.clear()
    collector_thread = threading.Thread(target=collect_frames, daemon=True)
    collector_thread.start()
    logger.info("스레드 시작됨")

def stop_collector():
    global collector_thread, stop_event
    stop_event.set()
    if collector_thread and collector_thread.is_alive():
        logger.info("데이터 수집 중단 요청 (스레드 종료 대기)")
        collector_thread.join(timeout=2)
    else:
        logger.info("데이터 수집 스레드 없음 / 이미 종료됨")


def open_capture(stream_url, last_log_time, retry_delay=1):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        now = time.time()
        if now - last_log_time > 60:
            logger.warning("RTMP 연결 실패, 재시도 대기 중...")
            last_log_time = now
        time.sleep(retry_delay)
        return None, last_log_time
    return cap, last_log_time


def handle_frame_fail(ret, frame, fail_count, last_log_time, max_fail=10, reconnect_fail=5):
    if ret and frame is not None:
        return False, 0, last_log_time
    fail_count += 1
    now = time.time()
    if fail_count > max_fail:
        if now - last_log_time > 60:
            logger.warning(f"프레임 읽기 실패 {fail_count}회")
            last_log_time = now
        if fail_count >= reconnect_fail:
            logger.warning("송출 끊김 -> 재연결 시도")
            return True, fail_count, last_log_time
        time.sleep(2)
    return False, fail_count, last_log_time


def run_inference(model, frame, conf=0.45):
    results = model.predict(frame, conf)
    boxes = results[0].boxes
    detected = len(boxes) > 0
    return detected, boxes


def check_event_condition(boxes, center_history, min_move=15, min_conf=0.5, aspect_thresh=2.0):
    if not boxes:
        return False, None
    best_box = max(boxes, key=lambda b: float(b.conf[0]))
    conf = float(best_box.conf[0])
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    cy = (y1 + y2) // 2
    center_history.append(cy)

    if len(center_history) >= 2:
        diffs = [abs(center_history[i] - center_history[i-1]) for i in range(1, len(center_history))]
        max_diff = max(diffs)
        if max_diff < min_move:
            logger.debug("정지 오탐 무시 (이동량 부족)")
            return False, None
        if conf < min_conf:
            logger.debug(f"conf 낮음 ({conf:.2f}) -> 무시")
            return False, None
        aspect_ratio = (y2 - y1) / ((x2 - x1) + 1e-6)
        if aspect_ratio < aspect_thresh:
            logger.debug(f"aspect ratio 낮음 ({aspect_ratio:.2f}) -> 무시")
            return False, None
        return True, best_box
    return False, None


def collect_event_frames(cap, frame_window, detect_window, frame, detected, post_frames=3):
    event_frames = []
    for f, det in zip(frame_window, detect_window):
        event_frames.append((f, det))
    event_frames.append((frame.copy(), detected))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    interval = 1.0 / fps
    last_frame = None
    for _ in range(post_frames):
        time.sleep(interval)
        ret2, frame2 = cap.read()
        if not ret2:
            break
        if last_frame is not None:
            diff = cv2.absdiff(last_frame, frame2)
            nonzero = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
            if nonzero < 5:
                continue
        event_frames.append((frame2.copy(), False))
        last_frame = frame2.copy()
    return event_frames


def log_fps(fps_count, fps_start, interval=10.0):
    fps_count += 1
    now_time = time.time()
    if now_time - fps_start >= interval:
        logger.info(f"현재 FPS: {fps_count / interval:.2f}")
        return 0, now_time
    return fps_count, fps_start


def upload_event(event_frames):
    """이벤트 프레임을 업로드 큐에 넣기"""
    upload_queue.put(event_frames)


def collect_frames():
    arrow_model = ArrowDetector()
    person_model = YOLO("yolov8n.pt")

    last_log_time = 0
    frame_count = 0
    frame_window = deque(maxlen=config.PRE_FRAMES)
    detect_window = deque(maxlen=config.PRE_FRAMES)
    center_history = deque(maxlen=5)

    last_event_time = 0
    fps_count = 0
    fps_start = time.time()

    while not stop_event.is_set():
        cap = None
        try:
            cap, last_log_time = open_capture(config.STREAM_URL, last_log_time)
            if cap is None:
                continue
            fail_count = 0

            while not stop_event.is_set():
                ret, frame = cap.read()
                reconnect, fail_count, last_log_time = handle_frame_fail(ret, frame, fail_count, last_log_time)
                if reconnect:
                    break
                if not ret or frame is None:
                    continue
                fail_count = 0
                frame_count += 1

                fps_count, fps_start = log_fps(fps_count, fps_start)

                detected, boxes = run_inference(arrow_model, frame)
                frame_window.append(frame.copy())
                detect_window.append(detected)

                now = time.time()
                if now - last_event_time < config.EVENT_COOL_DOWN:
                    continue

                ok, best_box = check_event_condition(boxes, center_history,
                                                     config.MIN_MOVE_FOR_START,
                                                     config.MIN_CONFIDENCE)
                if not ok:
                    continue

                person_results = person_model.predict(frame, conf=0.4, verbose=False, classes=[0])
                if len(person_results[0].boxes) > 0:
                    logger.debug("사람 감지됨 -> 이벤트 무시")
                    continue

                event_frames = collect_event_frames(cap, frame_window, detect_window,
                                                    frame, detected, config.POST_FRAMES)
                upload_event(event_frames)  # === 업로드는 큐로 넘김 ===
                last_event_time = now

        except Exception as e:
            now = time.time()
            if now - last_log_time > 60:
                logger.error("collect_frames() 예외:", e)
                last_log_time = now
            time.sleep(5)

        finally:
            if cap:
                cap.release()
            now = time.time()
            if now - last_log_time > 60:
                logger.info("세션 종료, 5초 후 재시작")
                last_log_time = now
            time.sleep(5)
