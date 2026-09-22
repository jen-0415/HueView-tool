"""
Phase 15 -- API endpoints.

Matches what frontend/src/api.js actually calls:
    POST /api/detect                    -> face check + crop preview
    POST /api/analyze                   -> returns {job_id}
    GET  /api/analyze/{job_id}/events   -> SSE: 6 "stage" events, then "result"
    GET  /api/health                    -> service + model status
    GET  /api/config                    -> SCC labels + region names
"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from .inference import (
    NoFaceDetected,
    STAGE_KEYS,
    SCC_LABELS,
    REGION_NAMES,
    run_detect,
    run_stage,
    build_result,
    models_loaded,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB -- per 15.3, stops a huge upload stalling the service

# In-memory job store. Fine for a single-process demo; if you ever run multiple
# workers, an EventSource request could land on a worker that doesn't know the
# job, so this would need Redis or similar.
JOBS: dict[str, dict] = {}


async def read_validated_image(image: UploadFile) -> bytes:
    """Per 15.3: reject non-images and oversized uploads before inference."""
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_type",
                    "message": "Please upload a JPG, JPEG, or PNG image."},
        )

    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "file_too_large",
                    "message": "Image is too large. Please upload a file under 10 MB."},
        )
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_file", "message": "The uploaded file is empty."},
        )
    return data


@router.post("/detect")
async def detect(image: UploadFile = File(...)):
    """Fast face check before committing to full inference."""
    data = await read_validated_image(image)

    try:
        # MTCNN is slow enough to block the event loop, so keep it off it.
        return await asyncio.to_thread(run_detect, data, image.filename or "")
    except NoFaceDetected as e:
        # api.js reads err?.detail?.message, so keep that shape
        raise HTTPException(
            status_code=422,
            detail={"code": "no_face_detected", "message": str(e)},
        )
    except Exception:
        log.exception("Detection failed")  # full trace server-side only
        raise HTTPException(
            status_code=500,
            detail={"code": "detection_failed",
                    "message": "Face detection failed. Please try another image."},
        )


@router.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    """Accepts the image, registers a job, returns its id. The actual work
    streams over /analyze/{job_id}/events."""
    data = await read_validated_image(image)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"image": data, "filename": image.filename or ""}
    return {"job_id": job_id}


@router.get("/analyze/{job_id}/events")
async def analyze_events(job_id: str):
    """SSE stream: one 'stage' event per pipeline stage (running, then done),
    followed by a single 'result' event carrying the full payload."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "unknown_job", "message": "That analysis job was not found."},
        )

    async def event_stream():
        try:
            # The real pipeline runs as one call, and it does its own face
            # detection -- so we don't call run_detect separately here. That
            # would be a third MTCNN pass on the same image (once in
            # /api/detect, once here, once inside classify_image), which is
            # several wasted seconds per upload on CPU.
            #
            # The stage beats stream first so the UI shows progress, then the
            # real work happens, then the result lands. Ordering is honest;
            # the per-stage millisecond figures are not individually measured
            # (see run_stage's docstring) -- the true cost is in each model's
            # inference_ms in the payload.
            for key in STAGE_KEYS:
                yield _sse("stage", {"key": key, "status": "running"})
                ms = await asyncio.to_thread(run_stage, key)
                yield _sse("stage", {"key": key, "status": "done", "ms": ms})

            payload = await asyncio.to_thread(build_result, job["image"], job["filename"])
            yield _sse("result", {"data": payload})

        except NoFaceDetected as e:
            yield _sse("error", {"code": "no_face_detected", "message": str(e)})
        except Exception:
            log.exception("Analysis failed for job %s", job_id)
            yield _sse("error", {"code": "analysis_failed",
                                 "message": "Analysis failed. Please try again."})
        finally:
            JOBS.pop(job_id, None)  # one-shot job; don't leak memory

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stops nginx buffering the stream if you deploy behind it
        },
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/health")
async def health():
    """Per 15.4 -- quick pre-demo sanity check."""
    loaded = models_loaded()
    return {
        "status": "ok",
        "models_loaded": loaded,
        "placeholder_mode": not all(loaded.values()),
    }


@router.get("/config")
async def config():
    """Per 15.4 -- lets the frontend avoid hardcoding label/region lists."""
    return {"scc_labels": SCC_LABELS, "regions": REGION_NAMES, "stages": STAGE_KEYS}
