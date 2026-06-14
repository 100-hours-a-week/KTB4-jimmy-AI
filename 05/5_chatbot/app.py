"""
FastAPI로 챗봇(다음 단어 생성 + 자기회귀 문장 생성)을 웹에서 사용할 수 있게 감싸기

실행 방법
    pip install fastapi uvicorn
    python train.py          # 먼저 학습해서 chatbot_model.pt 생성
    uvicorn app:app --reload

테스트
    curl "http://127.0.0.1:8000/generate?prompt=나는&mode=greedy"

    또는 브라우저에서 http://127.0.0.1:8000/docs 로 접속하면
    자동 생성된 Swagger UI에서 바로 테스트할 수 있다.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from generate import generate_text, load_model

app = FastAPI(title="토이 챗봇 API")

# 서버 시작 시 한 번만 모델을 로드해서 재사용 (요청마다 다시 불러오면 느림)
model, vocab = load_model()


class GenerateRequest(BaseModel):
    prompt: str
    mode: str = "greedy"        # "greedy" 또는 "sample"
    max_len: int = 15
    temperature: float = 1.0


@app.get("/")
def root():
    return {"message": "토이 챗봇 API. /docs 에서 사용법 확인 가능."}


@app.get("/generate")
def generate_get(prompt: str, mode: str = "greedy", max_len: int = 15, temperature: float = 1.0):
    """
    GET 방식: 쿼리 파라미터로 prompt 전달
    예) /generate?prompt=나는&mode=greedy
    """
    text = generate_text(model, vocab, prompt, max_len=max_len, mode=mode, temperature=temperature)
    return {"prompt": prompt, "generated": text}


@app.post("/generate")
def generate_post(req: GenerateRequest):
    """
    POST 방식: JSON body로 prompt 전달
    예) {"prompt": "나는", "mode": "sample", "temperature": 0.8}
    """
    text = generate_text(
        model, vocab, req.prompt,
        max_len=req.max_len, mode=req.mode, temperature=req.temperature,
    )
    return {"prompt": req.prompt, "generated": text}
