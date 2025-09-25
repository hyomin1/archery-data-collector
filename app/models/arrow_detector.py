from ultralytics import YOLO
from app.core import config

# YOLO 기반 화살 탐지 모델
class ArrowDetector:
     
    def __init__(self):
        self.model = YOLO(config.ARROW_MODEL_PATH)
        self.model.to("cuda")
        

    def predict(self,frame, conf=0.55, iou=0.4):
        results = self.model(frame, verbose=False, conf=conf, iou=iou)
        return results