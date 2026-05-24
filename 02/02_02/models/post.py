from pydantic import BaseModel, ConfigDict
from typing import List
from models.comment import Comment


class PostCreate(BaseModel):
    title: str
    content: str


class Post(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    content: str
    likes: int = 0
    comments: List[Comment] = []
