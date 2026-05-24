import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# 비밀번호에 @, # 등 특수문자가 있어도 URL 파싱이 깨지지 않도록 인코딩
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
DB_URL = f"mysql+pymysql://root:{DB_PASSWORD}@localhost:3306/board_db"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# 라우터에서 DB 세션을 주입받기 위한 의존성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
