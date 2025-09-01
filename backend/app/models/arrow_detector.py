from ultralytics import YOLO
from app import config

# YOLO 기반 화살 탐지 모델
class ArrowDetector:
     

    def __init__(self):
        self.model = YOLO(config.ARROW_MODEL_PATH)
        self.model.to("cuda")

    def predict(self,frame):
        results = self.model(frame, verbose=False)
        return results