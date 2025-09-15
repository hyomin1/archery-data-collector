from ultralytics import YOLO
from app import config

# YOLO 기반 과녁 탐지 모델
class TargetDetector:
    def __init__(self):
        self.model = YOLO(config.TARGET_MODEL_PATH)
        self.model.to("cuda")

    def predict(self,frame):
        return self.model(frame, verbose=False, conf=0.85)[0]