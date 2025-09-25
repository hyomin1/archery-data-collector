from app.services.collector import manager 
import time

def dummy_collect_frames(cam_id,url,stop_event):
    print(f"[{cam_id}] 수집 스레드 시작됨: {url}")
    count = 0
    while not stop_event.is_set() and count < 3:
        print(f"[{cam_id}] 수집 중: {count}회")
        time.sleep(1)
        count+=1

    print(f"[{cam_id}] 수집 스레드 종료됨")

def test_start_stop_collectors():

    manager.collect_frames = dummy_collect_frames


    manager.config.CAMERA_URLS = {
        "cam1": "rtmp://dummy1",
        "cam2": "rtmp://dummy2",
    }

    manager.start_all_collectors()
    time.sleep(3)
    manager.stop_all_collectors()
    for t in manager.collector_threads.values():
        assert not t.is_alive()