from contextlib import asynccontextmanager

from fastapi import FastAPI

from model.init__db import initialize_database
from view.view import configure_views


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="Regla Medidora", lifespan=lifespan)
configure_views(app)