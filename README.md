# archery-data-collector

국궁 화살 **데이터 수집용 시스템** (FastAPI + YOLOv8)

RTMP 영상 스트림에서 화살 감지 이벤트를 자동으로 감지하여, 감지 시점 전후 2초 구간의 영상 클립을 생성하고 저장하는 데이터 수집 시스템입니다.

수집된 데이터는 후속 단계에서 **라벨링 → 모델 재학습**에 활용됩니다.

---

## FastAPI 실행 방법

1. 가상환경 활성화

   ```bash
   venv\Scripts\activate   # (Windows)
   source venv/bin/activate  # (Linux/Mac)
   ```

2. 서버 실행

   ```bash
   uvicorn main:app --reload
   ```

   - 기본 주소: `http://127.0.0.1:8000`
   - 모니터링(확인용): `http://127.0.0.1:8000/rtmp`

---

## 패키지 관리

### 새 패키지 설치 시

```bash
pip install <패키지명>
pip freeze > requirements.txt
```

### 프로젝트 클론 후 패키지 설치

```bash
pip install -r requirements.txt
```
