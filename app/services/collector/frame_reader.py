import cv2, time
from app.core.logger import get_logger

logger = get_logger(__name__)

class FrameReader:
    def __init__(self,stream_url,reconnect_delay=5):
        self.stream_url = stream_url
        self.reconnect_delay = reconnect_delay
        self.cap = None
    
    def _open(self):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            logger.error(f"[FrameReader] 스트림 연결 실패: {self.stream_url}")
            self.cap = None
            return False
        logger.info(f"[FrameReader] 스트림 연결 완료: {self.stream_url}")
        return True
    
    def read(self):
        if self.cap is None or not self.cap.isOpened():
            if not self._open():
                time.sleep(self.reconnect_delay)
                return None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("[FrameReader] 프레임 읽기 실패")
            self._open()
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None        