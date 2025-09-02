import cv2
import time
import datetime
import threading
import logging
import os
from collections import deque
from app import config
from app.models.arrow_detector import ArrowDetector
from logging.handlers import RotatingFileHandler


# === 로깅 설정 ===
logging.basicConfig(
    level=logging.INFO,   # 기본 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL 가능), 운영 에서 WARNING으로 변경
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        RotatingFileHandler(
            "collector.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
    ]
)
logger = logging.getLogger(__name__)

collector_thread = None # 수집 스레드 
stop_event = threading.Event() # 수집 중단 신호
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
    global collector_thread,stop_event
    stop_event.set()
    
    if collector_thread and collector_thread.is_alive():
        logger.info("데이터 수집 중단 요청 (스레드 종료 대기)")
        collector_thread.join(timeout=2)
    else:
        logger.info("데이터 수집 스레드 없음 / 이미 종료됨")

def open_capture(stream_url, last_log_time, retry_delay=1):
    """스트림 URL을 열고 실패 시 None 반환"""
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        now = time.time()
        if now - last_log_time > 60:  # 로그는 60초 간격으로만 출력
            logger.warning("RTMP 연결 실패, 재시도 대기 중...")
            last_log_time = now
        time.sleep(retry_delay)
        return None, last_log_time
    return cap, last_log_time

def handle_frame_fail(ret, frame, fail_count, last_log_time, max_fail=10, reconnect_fail=5):
    """프레임 읽기 실패 처리"""
    if ret and frame is not None:
        return False, 0, last_log_time  # 실패 아님 → 정상 프레임

    fail_count += 1
    now = time.time()

    if fail_count > max_fail:
        if now - last_log_time > 60:
            logger.warning(f"프레임 읽기 실패 {fail_count}회")
            last_log_time = now
        if fail_count >= reconnect_fail:
            logger.warning("송출 끊김 -> 재연결 시도")
            return True, fail_count, last_log_time  # 재연결 필요
        time.sleep(2)
    return False, fail_count, last_log_time


def run_inference(model, frame, conf=0.45):
    # === YOLO 추론 ===
    results = model.predict(frame,conf)
    boxes = results[0].boxes
    detected = len(boxes) > 0
    return detected, boxes

def check_event_condition(boxes, center_history, min_move=15, min_conf=0.5, aspect_thresh=2.0):
    """탐지 결과가 이벤트(화살 진입) 조건을 만족하는지 검사"""
    if not boxes:
        return False, None

    # confidence 가장 높은 박스 선택
    best_box = max(boxes, key=lambda b: float(b.conf[0]))
    conf = float(best_box.conf[0])
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    cy = (y1 + y2) // 2
    center_history.append(cy)

    if len(center_history) >= 2:
        diffs = [abs(center_history[i] - center_history[i-1]) 
                 for i in range(1, len(center_history))]
        max_diff = max(diffs)

        # --- 오탐 필터 ---
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
    """이벤트 발생 시 앞/뒤 프레임까지 모아서 반환"""
    event_frames = []

    # 앞 프레임
    for f, det in zip(frame_window, detect_window):
        event_frames.append((f, det))

    # 현재 프레임
    event_frames.append((frame.copy(), detected))

    # 뒤 프레임
    for _ in range(post_frames):
        ret2, frame2 = cap.read()
        if not ret2:
            break
        event_frames.append((frame2.copy(), False))

    return event_frames

def log_fps(fps_count, fps_start, interval=10.0):
    """FPS 계산 및 1초마다 출력"""
    fps_count += 1
    now_time = time.time()

    if now_time - fps_start >= interval:
        logger.info(f"현재 FPS: {fps_count / interval:.2f}")
        return 0, now_time  

    return fps_count, fps_start


def save_event(event_frames):
    """이벤트 프레임 저장"""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")

    date_dir = os.path.join(config.DATA_DIR, date_str)
    os.makedirs(date_dir, exist_ok=True)

    event_dir = os.path.join(date_dir, f"event_{time_str}")
    os.makedirs(event_dir, exist_ok=True)

    for idx, (f, det) in enumerate(event_frames):
        status = "yolo_detected" if det else "undetected"
        filename = os.path.join(event_dir, f"frame_{idx}_{status}.jpg")
        cv2.imwrite(filename, f)

    logger.info(f"이벤트 감지 -> {len(event_frames)}프레임 저장 완료 ({event_dir})")

def collect_frames():
    arrow_model = ArrowDetector()

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

                # 최근 프레임 저장 (앞 프레임 확보용)
                frame_window.append(frame.copy())
                detect_window.append(detected)

                # 쿨다운 중이면 건너뛰기
                now = time.time()
                if now - last_event_time < config.EVENT_COOL_DOWN:
                    continue
                
                ok, best_box = check_event_condition(boxes,center_history,config.MIN_MOVE_FOR_START,config.MIN_CONFIDENCE)
                if not ok:
                    continue

                event_frames = collect_event_frames(cap,frame_window,detect_window,frame,detected,config.POST_FRAMES)

                save_event(event_frames)
                last_event_time = now  # 쿨다운 갱신

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
