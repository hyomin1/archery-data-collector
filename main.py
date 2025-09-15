from fastapi import FastAPI
from app.routes import endpoints
from app.services.collector import start_collector, stop_collector

app = FastAPI(title="국궁 프로젝트")

app.include_router(endpoints.router)


@app.on_event("startup")
def startup_event():
    start_collector()


@app.on_event("shutdown")
def shutdown_event():
    stop_collector()
