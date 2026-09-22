"""
Phase 14.8 -- classify_image()

Top-level orchestration. The API calls this and nothing else.

    preprocess_image()      -> crop + detection, or NoFaceDetected
    classify_illumination() -> shared by both models, computed once
    run_baseline()          -> global RGB path
    run_hueview()           -> regional SSR/CIELAB path

Response shape matches the frontend's mockData.js exactly, so switching
VITE_USE_MOCK to false changes nothing in the UI.
"""

from __future__ import annotations

from typing import Dict, Optional

from .preprocess import preprocess_image, NoFaceDetected
from .illumination import classify_illumination
from .baseline import run_baseline
from .hueview import run_hueview

# Re-exported so the API layer can catch it without importing preprocess.
__all__ = ["classify_image", "NoFaceDetected"]

SCC_INDEX = {f"SCC-{i}": i for i in range(1, 7)}


def _compare(baseline: Dict, hueview: Dict) -> Dict:
    """Per-image agreement between the two pipelines. Stays mostly null
    while SCC is placeholder -- the frontend renders that fine."""
    b_scc, h_scc = baseline.get("scc"), hueview.get("scc")
    b_conf, h_conf = baseline.get("confidence"), hueview.get("confidence")

    agreement: Optional[bool] = None
    scc_delta: Optional[int] = None
    if b_scc and h_scc:
        agreement = b_scc == h_scc
        scc_delta = abs(SCC_INDEX.get(b_scc, 0) - SCC_INDEX.get(h_scc, 0))

    confidence_delta: Optional[float] = None
    if b_conf is not None and h_conf is not None:
        confidence_delta = round(abs(b_conf - h_conf), 4)

    return {
        "agreement": agreement,
        "scc_delta": scc_delta,
        "confidence_delta": confidence_delta,
    }


def classify_image(image_bytes: bytes, filename: str = "") -> Dict:
    """
    One image in, the full dual-model result out.

    Raises NoFaceDetected -- the API turns that into a 422.
    """
    crop, detection = preprocess_image(image_bytes)

    illumination = classify_illumination(crop)
    baseline = run_baseline(crop)
    hueview = run_hueview(crop)

    return {
        "image": {"width": 224, "height": 224, "preview": None},
        "detection": detection,
        "illumination": illumination,
        "models": {"baseline": baseline, "hueview": hueview},
        "comparison": _compare(baseline, hueview),
        "test_set_metrics": {
            "available": False,
            "source": "Phase 11 held-out test split",
        },
    }
