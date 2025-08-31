from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from collections import deque
from dotenv import load_dotenv
from models.corner_regressor import CornerRegressor
from torchvision import transforms

import cv2
import time
import threading
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

app = FastAPI()


arrow_model = YOLO("weights/best2.pt")
arrow_model.to("cuda")

target_model = YOLO("weights/target.pt")  # 과녁
target_model.to("cuda")

corner_model = CornerRegressor()
corner_model.load_state_dict(
    torch.load("weights/corner_regressor.pt", map_location="cpu")
)
corner_model.eval()

transform = transforms.ToTensor()

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

    event_cooldown = 5  # 이벤트 최소 간격
    last_event_time = 0  # 마지막 이벤트 발생 시각

    buffer = deque()  
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
                    results = arrow_model(frame, verbose=False)
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
# 전역 변수: 과녁 코너 좌표 캐싱
target_corners = None
corner_lock = threading.Lock()

def get_target_corners(frame):
    """과녁 코너를 최초 1회만 CornerRegressor로 추정"""
    global target_corners
    with corner_lock:
        if target_corners is None:  # 아직 추정 안 한 경우
            frame_resized = cv2.resize(frame, (128, 128))
            tensor = transform(frame_resized).unsqueeze(0)
            with torch.no_grad():
                preds = corner_model(tensor).cpu().numpy()[0]
            h, w, _ = frame.shape
            target_corners = [
                (int(preds[i] * w), int(preds[i+1] * h))
                for i in range(0, 8, 2)
            ]
            print("과녁 코너 좌표 고정:", target_corners)
    return target_corners


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
                # --------------------
                # 1) 과녁 탐지 + Corner CNN
                # --------------------
                t_results = target_model(frame, verbose=False)
                for box in t_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # crop → CNN
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_resized = cv2.resize(crop, (128, 128))
                    tensor = transform(crop_resized).unsqueeze(0)

                    with torch.no_grad():
                        preds = corner_model(tensor).cpu().numpy()[0]

                    # bbox 기준 좌표 복원
                    corners = []
                    for i in range(0, 8, 2):
                        cx = int(x1 + preds[i]   * (x2 - x1))
                        cy = int(y1 + preds[i+1] * (y2 - y1))
                        corners.append((cx, cy))

                    # 빨간 테두리
                    for pt in corners:
                        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                    for j in range(4):
                        cv2.line(frame, corners[j], corners[(j+1)%4], (0,0,255), 2)

                # --------------------
                # 2) 화살 탐지
                # --------------------
                a_results = arrow_model(frame, verbose=False)
                for box in a_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

            except Exception as e:
                print("추론 에러:", e)

            # 스트리밍 전송
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
