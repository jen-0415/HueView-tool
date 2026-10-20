"""
Phase 15 -- FastAPI entrypoint.

Run from ml_pipeline/:
    uvicorn src.api.main:app --port 8000
or from the repo root:
    uvicorn ml_pipeline.src.api.main:app --port 8000
Add --reload during development so code changes are picked up.

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
from ..inference.models import load_models, models_loaded

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Per 14.2 / 15.1: load models ONCE at startup, never per request.
    load_models()
    status = models_loaded()
    if status["baseline"]:
        log.info("Baseline model loaded and ready.")
    else:
        log.warning("Baseline weights missing -- /api/analyze will return placeholder SCC for baseline.")
    if status["hueview"]:
        log.info("ChroLumi model loaded and ready.")
    else:
        log.warning("ChroLumi weights missing -- /api/analyze will return placeholder SCC for ChroLumi.")
    yield
    log.info("Shutting down ChroLumi API")


app = FastAPI(
    title="ChroLumi API",
    description="Dual-pipeline skin tone classification: Baseline vs. ChroLumi",
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
    return {"service": "ChroLumi API", "docs": "/docs", "health": "/api/health"}