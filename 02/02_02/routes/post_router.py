from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.post import Post, PostCreate
from controllers import post_controller
from database import get_db

router = APIRouter()


@router.post("/posts", response_model=Post)
def create_post(post_data: PostCreate, db: Session = Depends(get_db)):
    return post_controller.create_post(db, post_data)


@router.get("/posts", response_model=list[Post])
def get_posts(db: Session = Depends(get_db)):
    return post_controller.get_posts(db)


@router.get("/posts/{post_id}", response_model=Post)
def get_post(post_id: int, db: Session = Depends(get_db)):
    post = post_controller.get_post(db, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다")
    return post


@router.patch("/posts/{post_id}", response_model=Post)
def like_post(post_id: int, db: Session = Depends(get_db)):
    post = post_controller.like_post(db, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다")
    return post
