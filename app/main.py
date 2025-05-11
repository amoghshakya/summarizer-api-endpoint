import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router

app = FastAPI()
app.mount("/audio", StaticFiles(directory="/tmp"), name="audio")
app.include_router(router)
