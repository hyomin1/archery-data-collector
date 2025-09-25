import time, cv2
from collections import deque
from ultralytics import YOLO
from app.models.arrow_detector import ArrowDetector
from app.services.collector.frame_reader import FrameReader
from app.services.collector.inference import run_inference, check_event_condition
from app.services.collector.uploader import upload_event
from app.core.logger import get_logger
from app.core import config
from datetime import datetime

logger = get_logger(__name__)


def collect_frames(cam_id, stream_url, stop_event):
    arrow_model = ArrowDetector()
    person_model = YOLO("yolov8n.pt")
    reader = FrameReader(stream_url)

    frame_window = deque(maxlen=config.PRE_FRAMES)
    detect_window = deque(maxlen=config.PRE_FRAMES)  # (det, boxes) 저장
    center_history = deque(maxlen=5)
    last_event_time = 0

    while not stop_event.is_set():
        frame = reader.read()
        if frame is None:
            continue

        # 1. 화살 추론만 먼저 실행
        detected, boxes = run_inference(arrow_model, frame)
        frame_window.append(frame.copy())
        detect_window.append((detected, boxes))  # ✅ boxes까지 저장

        now = time.time()
        if now - last_event_time < config.EVENT_COOL_DOWN:
            continue

        # 2. 이벤트 조건 확인
        ok, best_box = check_event_condition(
            boxes, center_history,
            config.MIN_MOVE_FOR_START,
            config.MIN_CONFIDENCE
        )
        if not ok:
            continue

        # 3. 이벤트 조건 충족 시에만 사람 탐지 실행
        person_results = person_model.predict(
            frame, conf=0.4, verbose=False, classes=[0]
        )
        if len(person_results[0].boxes) > 0:
            logger.debug(f"[{cam_id}] 사람 감지 → 이벤트 무시")
            continue

        # 4. 업로드 (사람 없을 때만)
        event_frames = collect_event_frames(
            reader.cap, frame_window, detect_window,
            frame, detected, boxes, config.POST_FRAMES,arrow_model
        )
        upload_event(cam_id, datetime.now(), event_frames)
        last_event_time = now

    reader.release()
    logger.info(f"[{cam_id}] 종료됨")


def collect_event_frames(cap, frame_window, detect_window, frame, detected, boxes, post_frames, arrow_model):
    """
    이벤트 발생 시 이전/현재/이후 프레임 묶음 수집
    """
    event_frames = []

    # 이전 프레임들
    for f, (det, bxs) in zip(frame_window, detect_window):
        event_frames.append((f, det, bxs))

    # 현재 프레임
    event_frames.append((frame.copy(), detected, boxes))

    # 이후 프레임 몇 장 추가
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    interval = 1.0 / fps
    last_frame = None
    for _ in range(post_frames):
        time.sleep(interval)
        ret2, frame2 = cap.read()
        if not ret2:
            break
        
        det2, boxes2 = run_inference(arrow_model, frame2)
        if last_frame is not None:
            diff = cv2.absdiff(last_frame, frame2)
            nonzero = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
            if nonzero < 5:  # 거의 변화 없으면 스킵
                continue
        event_frames.append((frame2.copy(), det2, boxes2))  # ✅ 형식 맞춤
        last_frame = frame2.copy()

    return event_frames
