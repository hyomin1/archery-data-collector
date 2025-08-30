from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from collections import deque
from dotenv import load_dotenv

import cv2
import time
import threading
import os

app = FastAPI()


model = YOLO("models/best2.pt")
model.to("cuda")

load_dotenv()  
STREAM_URL = os.getenv("STREAM_URL")


os.makedirs("data/events", exist_ok=True)

# shutdown clean-up용
stop_collect = False  


# -------------------------------
#  데이터 수집 루프
# -------------------------------
from collections import deque
import datetime

def collect_frames():
    global stop_collect
    last_log_time = 0

    pre_seconds = 2   # 감지 전 2초
    post_seconds = 2  # 감지 후 2초

    event_cooldown = 5  # 이벤트 최소 간격(초)
    last_event_time = 0  # 마지막 이벤트 발생 시각

    buffer = deque()  # FPS 읽기 전까지 maxlen 설정 안함
    saving_post = 0
    save_queue = []

    while not stop_collect:
        cap = None
        try:
            cap = cv2.VideoCapture(STREAM_URL)
            if not cap.isOpened():
                now = time.time()
                if now - last_log_time > 60:
                    print("RTMP 연결 실패, 10초 후 재시도")
                    last_log_time = now
                time.sleep(10)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 30  # fallback
            fps = int(fps)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            pre_frames = fps * pre_seconds
            post_frames = fps * post_seconds
            buffer = deque(maxlen=pre_frames)

            fail_count = 0

            while not stop_collect:
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    now = time.time()
                    if now - last_log_time > 60:
                        print(f"프레임 읽기 실패 {fail_count}회")
                        last_log_time = now
                    if fail_count >= 5:
                        print("송출 끊김 → 재연결 시도")
                        break
                    time.sleep(2)
                    continue

                fail_count = 0
                buffer.append(frame.copy())

                try:
                    results = model(frame, verbose=False)
                    if len(results[0].boxes) > 0 and saving_post == 0:
                        now = time.time()
                        if now - last_event_time < event_cooldown:
                            continue
                        last_event_time = now
                        # 이벤트 발생 → 버퍼 + 이후 프레임 저장
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"data/events/event_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

                        # 과거 프레임 저장
                        for f in buffer:
                            out.write(f)

                        # 현재 프레임 저장
                        out.write(frame)

                        save_queue = [out, filename]
                        saving_post = post_frames
                        print(f"이벤트 감지 → 클립 저장 시작: {filename}")

                    elif saving_post > 0:
                        # 이후 프레임 저장
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




# -------------------------------
#  모니터링 스트리밍
# -------------------------------
def generate_frames():
    cap = None
    try:
        cap = cv2.VideoCapture(STREAM_URL)
        if not cap.isOpened():
            print("RTMP 연결 실패 (모니터링 불가)")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = model(frame, verbose=False)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            except Exception as e:
                print("YOLO 추론 에러(모니터링):", e)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    finally:
        if cap:
            cap.release()
        print("모니터링 세션 종료")


# -------------------------------
#  FastAPI 엔드포인트
# -------------------------------
@app.get("/")
def root():
    return {"message": "국궁 프로젝트"}


@app.get("/rtmp")
def detect_rtmp():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# -------------------------------
#  FastAPI lifecycle
# -------------------------------
@app.on_event("startup")
def startup_event():
    global stop_collect
    stop_collect = False
    threading.Thread(target=collect_frames, daemon=True).start()
    print("데이터 수집 스레드 시작")


@app.on_event("shutdown")
def shutdown_event():
    global stop_collect
    stop_collect = True
    print("서버 종료 → 데이터 수집 중단")
