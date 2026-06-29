from fastapi import FastAPI
from pydantic import BaseModel
from graph import app as app_graph
# fastapi
app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/query")
def query(request: Query):

    return {"answer": app_graph.invoke({"question": request.prompt})["answer"]}