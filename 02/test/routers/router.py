from fastapi import APIRouter
import controllers.controller as ctrl

router=APIRouter()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.get("/posts")
def read_posts():
    return ctrl.get_posts()

@router.get("/posts/post/{post_id}")
def read_post(post_id):
    return ctrl.get_post(post_id)

@router.patch("/posts/post/{post_id}/like")
def like_post(post_id):
    return ctrl.like_post(post_id)