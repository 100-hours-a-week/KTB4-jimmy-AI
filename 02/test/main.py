from fastapi import FastAPI
from routers import router

app = FastAPI()  #fastapi instance 만들기

app.include_router(router.router)