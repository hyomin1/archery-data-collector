from collections import deque
from app.core.logger import get_logger
logger = get_logger(__name__)

def run_inference(model, frame, conf=0.45):
    results = model.predict(frame, conf=conf)
    boxes = results[0].boxes

    if len(boxes) > 1:
        best_box = max(boxes, key=lambda b: float(b.conf[0]))
        boxes = [best_box]
        
    detected = len(boxes) > 0
    return detected, boxes


def check_event_condition(boxes, center_history: deque, min_move=15, min_conf=0.5, aspect_thresh=2.0):
    if not boxes:
        return False, None

    best_box = max(boxes, key=lambda b: float(b.conf[0]))
    conf = float(best_box.conf[0])
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    cy = (y1 + y2) // 2
    center_history.append(cy)

    if len(center_history) >= 2:
        diffs = [abs(center_history[i] - center_history[i-1]) for i in range(1, len(center_history))]
        if max(diffs) < min_move or conf < min_conf:
            return False, None
        aspect_ratio = (y2 - y1) / ((x2 - x1) + 1e-6)
        if aspect_ratio < aspect_thresh:
            return False, None
        return True, best_box
    return False, None