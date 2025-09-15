import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# 기본 환경 변수
# -------------------------------
STREAM_URL = os.getenv("STREAM_URL")

# 데이터 저장 폴더
DATA_DIR=os.getenv("DATA_DIR","data/events")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------
# 수집기 파라미터
# -------------------------------
PRE_SECONDS =  int(os.getenv("PRE_SECONDS", 2))
POST_SECONDS = int(os.getenv("POST_SECONDS", 2))
EVENT_COOL_DOWN = int(os.getenv("EVENT_COOLDOWN", 5))

# -------------------------------
# 모델 경로
# -------------------------------
ARROW_MODEL_PATH: str = os.getenv("ARROW_MODEL_PATH", "weights/best2.pt")
TARGET_MODEL_PATH: str = os.getenv("TARGET_MODEL_PATH", "weights/target.pt")
CORNER_MODEL_PATH: str = os.getenv("CORNER_MODEL_PATH", "weights/corner_regressor2.pt")

# 이벤트 감지 관련 상수
PRE_FRAMES = 4          # 이벤트 발생 전 확보할 프레임 수
POST_FRAMES = 4         # 이벤트 발생 후 확보할 프레임 수
MIN_MOVE_FOR_START = 15 # 이벤트 시작 최소 이동량 (픽셀)
MIN_CONFIDENCE = 0.5    # 이벤트 시작 최소 confidence
EVENT_COOL_DOWN = 3     # 이벤트 쿨다운 시간 (초)