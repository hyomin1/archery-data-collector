import cv2
import time
import datetime
import threading
from collections import deque
from app import config

from app.models.arrow_detector import ArrowDetector

stop_collect = False
collector_thread = None

def start_collector():
    global collector_thread, collector_thread
    collector_thread = threading.Thread(target=collect_frames, daemon=True)
    collector_thread.start()

def stop_collector():
    global collector_thread, stop_collect
    stop_collect = True
    
    if collector_thread and collector_thread.is_alive():
        print("데이터 수집 중단 요청 (스레드 종료 대기)")
    else:
        print("데이터 수집 스레드 없음 / 이미 종료됨")


def collect_frames():

    global stop_collect
    arrow_model = ArrowDetector()

    last_log_time = 0
    last_event_time = 0  
    saving_post = 0
    save_queue = []
    buffer = deque() 

    while not stop_collect:
        cap = None
        try:
            cap = cv2.VideoCapture(config.STREAM_URL)
            if not cap.isOpened():
                now = time.time()
                if now - last_log_time > 60:
                    print("RTMP 연결 실패, 10초 후 재시도")
                    last_log_time = now
                time.sleep(10)
                continue

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            pre_frames = fps * config.PRE_SECONDS
            post_frames = fps * config.POST_SECONDS
            buffer = deque(maxlen=pre_frames)

            fail_count = 0

            while not stop_collect:
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    if fail_count > 10:
                        now = time.time()
                        if now - last_log_time > 60:
                            print(f"프레임 읽기 실패 {fail_count}회")
                            last_log_time = now
                        if fail_count >= 5:
                            print("송출 끊김 -> 재연결 시도")
                            break
                        time.sleep(2)
                        continue
                fail_count = 0
                buffer.append(frame.copy())

                try:
                    results = arrow_model.predict(frame)
                    if len(results[0].boxes) > 0 and saving_post == 0:
                        now = time.time()
                        if now - last_event_time < config.EVENT_COOLDOWN:
                            continue
                        last_event_time = now

                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{config.DATA_DIR}/event_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

                        # 버퍼된 과거 프레임 저장
                        for f in buffer:
                            out.write(f)
                         # 현재 프레임도 저장
                        out.write(frame)

                        save_queue = [out, filename]
                        saving_post = post_frames
                        print(f"이벤트 감지 -> 클립 저장 시작: {filename}")

                    elif saving_post > 0:
                        out, filename = save_queue
                        out.write(frame)
                        saving_post -= 1
                        if saving_post == 0:
                            out.release()
                            print(f"클립 저장 완료: {filename}")
                            save_queue = []

                except Exception as e:
                    now = time.time()
                    if now - last_log_time > 60:
                        print("YOLO 추론 에러:", e)
                        last_log_time = now

        except Exception as e:
            now = time.time()
            if now - last_log_time > 60:
                print("collect_frames() 예외:", e)
                last_log_time = now
            time.sleep(5)

        finally:
            if cap:
                cap.release()
            now = time.time()
            if now - last_log_time > 60:
                print("세션 종료, 5초 후 재시작")
                last_log_time = now
            time.sleep(5)