from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum

app = FastAPI()  #fastapi instance 만들기

@app.get("/")
def read_root():
    return {"Hello": "World"}