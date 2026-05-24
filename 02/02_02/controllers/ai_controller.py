import httpx
from typing import Optional
from sqlalchemy.orm import Session
from orm import PostORM

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:latest"


async def summarize_post(db: Session, post_id: int) -> Optional[str]:
    # DB에서 게시글 조회
    orm_post = db.query(PostORM).filter(PostORM.id == post_id).first()
    if orm_post is None:
        return None

    prompt = (
        f"다음 게시글을 한국어로 두세 문장으로 요약해줘.\n\n"
        f"제목: {orm_post.title}\n"
        f"내용: {orm_post.content}\n\n"
        f"요약:"
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"]
