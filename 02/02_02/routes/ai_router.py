import httpx
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from controllers import ai_controller
from database import get_db

router = APIRouter()


@router.get("/posts/{post_id}/summary")
async def summarize_post(post_id: int, db: Session = Depends(get_db)):
    try:
        summary = await ai_controller.summarize_post(db, post_id)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama 서버에 연결할 수 없습니다")

    if summary is None:
        raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다")

    return {"post_id": post_id, "summary": summary}
