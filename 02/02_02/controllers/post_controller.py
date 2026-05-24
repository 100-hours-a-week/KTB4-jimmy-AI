from typing import Optional, List
from sqlalchemy.orm import Session
from models.post import Post, PostCreate
from orm import PostORM


def create_post(db: Session, post_data: PostCreate) -> Post:
    orm_post = PostORM(title=post_data.title, content=post_data.content)
    db.add(orm_post)
    db.commit()
    db.refresh(orm_post)
    return Post.model_validate(orm_post)


def get_posts(db: Session) -> List[Post]:
    orm_posts = db.query(PostORM).all()
    return [Post.model_validate(p) for p in orm_posts]


def get_post(db: Session, post_id: int) -> Optional[Post]:
    orm_post = db.query(PostORM).filter(PostORM.id == post_id).first()
    if orm_post is None:
        return None
    return Post.model_validate(orm_post)


def like_post(db: Session, post_id: int) -> Optional[Post]:
    orm_post = db.query(PostORM).filter(PostORM.id == post_id).first()
    if orm_post is None:
        return None
    orm_post.likes += 1
    db.commit()
    db.refresh(orm_post)
    return Post.model_validate(orm_post)
