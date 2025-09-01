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
EVENT_COOLDOWN = int(os.getenv("EVENT_COOLDOWN", 5))

# -------------------------------
# 모델 경로
# -------------------------------
ARROW_MODEL_PATH: str = os.getenv("ARROW_MODEL_PATH", "weights/best2.pt")
TARGET_MODEL_PATH: str = os.getenv("TARGET_MODEL_PATH", "weights/target.pt")
CORNER_MODEL_PATH: str = os.getenv("CORNER_MODEL_PATH", "weights/corner_regressor2.pt")