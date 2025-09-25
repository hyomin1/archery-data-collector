from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routes import endpoints
from app.services.collector.manager import start_all_collectors, stop_all_collectors


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    start_all_collectors()
    print("[INFO] 모든 카메라 Collector 시작됨")

    yield  # 👈 여기가 앱 실행 구간

    # shutdown
    stop_all_collectors()
    print("[INFO] 모든 카메라 Collector 종료됨")


app = FastAPI(title="국궁 자동화 데이터 수집기", lifespan=lifespan)

# 라우터 등록
app.include_router(endpoints.router)
