"""
Phase 15 -- FastAPI entrypoint.

Run from the project root:
    uvicorn src.api.main:app --reload --port 8000

Interactive docs at http://localhost:8000/docs -- easiest way to test every
endpoint without writing curl commands.

The frontend defaults to http://localhost:8000 (api.js BASE), so port 8000
matters. To point the frontend at this instead of its mock data, set
VITE_USE_MOCK=false in the frontend's .env.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Per 14.2 / 15.1: load models ONCE at startup, never per request.
    # When Phase 14 is ready, call load_models() here and stash the result
    # somewhere inference.py can reach.
    log.info("Starting HueView API (placeholder mode -- no real weights loaded yet)")
    yield
    log.info("Shutting down HueView API")


app = FastAPI(
    title="HueView API",
    description="Dual-pipeline skin tone classification: Baseline vs. HueView",
    version="0.1.0",
    lifespan=lifespan,
)

# Vite dev server runs on 5173; without this the browser blocks every request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"service": "HueView API", "docs": "/docs", "health": "/api/health"}
