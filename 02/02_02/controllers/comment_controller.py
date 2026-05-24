from typing import Optional
from sqlalchemy.orm import Session
from models.comment import Comment, CommentCreate
from orm import CommentORM, PostORM


def create_comment(db: Session, post_id: int, comment_data: CommentCreate) -> Optional[Comment]:
    orm_post = db.query(PostORM).filter(PostORM.id == post_id).first()
    if orm_post is None:
        return None

    orm_comment = CommentORM(content=comment_data.content, post_id=post_id)
    db.add(orm_comment)
    db.commit()
    db.refresh(orm_comment)
    return Comment.model_validate(orm_comment)
