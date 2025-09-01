from fastapi import FastAPI
from app.routes import endpoints
from app.services.collector import start_collector, stop_collector

app = FastAPI(title="국궁 프로젝트")

app.include_router(endpoints.router)


@app.on_event("startup")
def startup_event():
    start_collector()
    print("서버 시작: 데이터 수집 스레드 실행")


@app.on_event("shutdown")
def shutdown_event():
    stop_collector()
    print("서버 종료: 데이터 수집 중단")
