from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.comment import Comment, CommentCreate
from controllers import comment_controller
from database import get_db

router = APIRouter()


@router.post("/posts/{post_id}/comments", response_model=Comment)
def create_comment(post_id: int, comment_data: CommentCreate, db: Session = Depends(get_db)):
    comment = comment_controller.create_comment(db, post_id, comment_data)
    if comment is None:
        raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다")
    return comment
