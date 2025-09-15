from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.streamer import generate_frames

router = APIRouter()


@router.get("/")
def root():
    """서버 상태 확인용 엔드포인트"""
    return {"message": "국궁 프로젝트"}


@router.get("/rtmp")
def detect_rtmp():
    """RTMP 스트림 실시간 모니터링"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
