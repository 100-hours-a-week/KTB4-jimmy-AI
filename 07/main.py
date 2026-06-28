from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask
# fastapi
app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/query")
def query(request: Query):

    return {"answer": ask(request.prompt)}