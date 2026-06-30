from fastapi import FastAPI
from pydantic import BaseModel
from graph import app as app_graph
# fastapi
app = FastAPI()

class Query(BaseModel):
    prompt: str
    top_k: int = 3
    limit: int = 4

@app.post("/query")
def query(request: Query):

    return {"answer": app_graph.invoke({"question": request.prompt, 
                                        "top_k": request.top_k,
                                        "limit": request.limit
                                        })["answer"]}