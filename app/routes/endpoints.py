from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.streamer import generate_frames
from app.core import config


router = APIRouter()


@router.get("/")
def root():
    """서버 상태 확인용 엔드포인트"""
    return {"message": "국궁 프로젝트"}


@router.get("/cam1")
def monitor_cam1():
    return StreamingResponse(
        generate_frames(config.CAMERA_URLS['target1'], cam_id=1),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/cam2")
def monitor_cam2():
    return StreamingResponse(
        generate_frames(config.CAMERA_URLS['target2'], cam_id=2),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/cam3")
def monitor_cam3():
    return StreamingResponse(
        generate_frames(config.CAMERA_URLS['target3'], cam_id=3),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

