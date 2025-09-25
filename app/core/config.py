import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# 기본 환경 변수
# -------------------------------
STREAM_URL = os.getenv("CAM1_URL")


CAMERA_URLS = {
    #'target1':os.getenv("CAM1_URL"),
    #'target2':os.getenv("CAM2_URL"),
    'target3':os.getenv("CAM3_URL")
}


STREAM_URLS = [
    #os.getenv("TEST"),
    os.getenv("CAM1_URL"),
    #os.getenv("CAM2_URL"),
    #os.getenv("CAM3_URL")
]
# 데이터 저장 폴더
DATA_DIR=os.getenv("DATA_DIR","data/events")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------
# 수집기 파라미터
# -------------------------------
PRE_SECONDS =  int(os.getenv("PRE_SECONDS", 4))
POST_SECONDS = int(os.getenv("POST_SECONDS", 4))
EVENT_COOL_DOWN = int(os.getenv("EVENT_COOLDOWN", 5))

# -------------------------------
# 모델 경로
# -------------------------------
ARROW_MODEL_PATH: str = os.getenv("ARROW_MODEL_PATH", "weights/best2.pt")
TARGET_MODEL_PATH: str = os.getenv("TARGET_MODEL_PATH", "weights/target.pt")
CORNER_MODEL_PATH: str = os.getenv("CORNER_MODEL_PATH", "weights/corner_regressor2.pt")


ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")

# 이벤트 감지 관련 상수
PRE_FRAMES = 4          # 이벤트 발생 전 확보할 프레임 수
POST_FRAMES = 4         # 이벤트 발생 후 확보할 프레임 수
MIN_MOVE_FOR_START = 15 # 이벤트 시작 최소 이동량 (픽셀)
MIN_CONFIDENCE = 0.55    # 이벤트 시작 최소 confidence
EVENT_COOL_DOWN = 3     # 이벤트 쿨다운 시간 (초)