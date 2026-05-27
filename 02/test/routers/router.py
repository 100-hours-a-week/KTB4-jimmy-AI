from fastapi import APIRouter

router=APIRouter()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.get("/posts")
def read_posts():
    return {"hello":"world"}

@router.get("/posts/post/{post_id}")
def read_post(post_id):
    id=post_id
    return {"hello":"world2"}