"""
Phase 15 -- inference wrapper.

The bridge between the API layer and Phase 14. routes.py imports from here
and never touches the pipeline directly, so the web service and the model
code stay independent of each other.

The placeholder data this file used to return is gone: run_detect() and
build_result() now call the real pipeline. Everything is real except SCC
classification, which stays null until trained weights exist -- the
frontend already handles that via each model's `placeholder` flag.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from ..inference.pipeline import classify_image
from ..inference.preprocess import preprocess_image, NoFaceDetected  # noqa: F401
from ..inference.illumination import classify_illumination
from ..inference.models import models_loaded as _models_loaded
from ..hueview.regions import REGION_DISPLAY, REGION_ORDER

# Matches constants.js STAGES -- the frontend won't advance its progress UI
# unless these exact keys come back in this order.
STAGE_KEYS = ["detect", "ssr", "segment", "cnn", "lab", "undertone"]

# Matches constants.js SCC order, and must equal Phase 10's label_order.json
# once weights land, or every prediction is mislabeled without erroring.
SCC_LABELS = ["SCC-1", "SCC-2", "SCC-3", "SCC-4", "SCC-5", "SCC-6"]

REGION_NAMES = [REGION_DISPLAY[r] for r in REGION_ORDER] + [REGION_DISPLAY["full_face"]]


def run_detect(image_bytes: bytes, filename: str = "") -> Dict:
    """
    POST /api/detect -- the fast face check before committing to full
    inference. Real MTCNN now, with Phase 2's exact crop and gates.
    """
    _crop, detection = preprocess_image(image_bytes)
    return {
        "confidence": detection["confidence"],
        "landmarks_found": detection["landmarks_found"],
        "bbox": detection["bbox"],
        "upscaled": detection["upscaled"],
        "crop_preview": None,  # TODO: base64-encode _crop if the UI wants it
    }


def run_stage(stage_key: str) -> int:
    """
    Stage timing for the SSE progress stream.

    The real pipeline runs as one call inside build_result() rather than as
    six separately-timed steps, so these are short beats that keep the
    progress UI honest about ORDER without inventing per-stage durations.
    The true end-to-end cost shows up in each model's inference_ms.
    """
    time.sleep(0.05)
    return 50


def build_result(image_bytes: bytes, filename: str = "") -> Dict:
    """The full dual-model payload, from the real pipeline."""
    return classify_image(image_bytes, filename)


def models_loaded() -> Dict[str, bool]:
    return _models_loaded()
