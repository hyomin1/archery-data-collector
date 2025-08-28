from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import datetime

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello Gukgung Project 🎯"}


@app.post("/uploadframe/")
async def upload_frame(file: UploadFile = File(...)):
    # 바이너리 → OpenCV 이미지
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    # 프레임 크기 확인 
    print("받은 프레임:", frame.shape)

    # 첫 프레임만 저장
    filename = f"frame_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"저장 완료: {filename}")
    # 추후 저장한 frame을 클라우드 저장소에 전달
    return {"status": "ok", "saved": filename}
