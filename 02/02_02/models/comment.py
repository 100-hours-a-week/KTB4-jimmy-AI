from pydantic import BaseModel, ConfigDict


class CommentCreate(BaseModel):
    content: str


class Comment(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    content: str
