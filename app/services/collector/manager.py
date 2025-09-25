import threading
from app.services.collector.worker import collect_frames
from app.core import config
from app.core.logger import get_logger


logger = get_logger(__name__)

collector_threads = {}
stop_events = {}

def start_all_collectors():
    """모든 카메라 수집 스레드 시작"""
    global collector_threads, stop_events

    for cam_id, url in config.CAMERA_URLS.items():
        if cam_id in collector_threads and collector_threads[cam_id].is_alive():
            logger.warning(f"[{cam_id}] 이미 실행 중")
            continue
        
        stop_events[cam_id] = threading.Event()
        t = threading.Thread(
            target=collect_frames, args=(cam_id, url, stop_events[cam_id]), daemon=True
        )
        collector_threads[cam_id] = t
        t.start()
        logger.info(f"[{cam_id}] 수집 스레드 시작됨: {url}")

def stop_all_collectors():
    for cam_id, ev in stop_events.items():
        ev.set()
        logger.info(f"[{cam_id}] 종료 요쳥됨")
    
    for cam_id, t in collector_threads.items():
        if t.is_alive():
            t.join(timeout=2)
            logger.info(f"[{cam_id}] Collector 종료 완료")
  
