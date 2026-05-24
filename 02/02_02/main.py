from fastapi import FastAPI
from routes import post_router, comment_router, ai_router
from database import engine, Base
import orm  # ORM 모델을 Base에 등록

# 서버 시작 시 테이블 자동 생성
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(post_router.router)
app.include_router(comment_router.router)
app.include_router(ai_router.router)
