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

import base64
import logging
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..inference.pipeline import classify_image
from ..inference.preprocess import preprocess_image, NoFaceDetected  # noqa: F401
from ..inference.hueview import compute_ssr
from ..inference.illumination import classify_illumination
from ..inference.models import models_loaded as _models_loaded
from ..hueview import ssr_normalization as ssr_mod
from ..hueview.regions import REGION_DISPLAY, REGION_ORDER

log = logging.getLogger(__name__)

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


def _to_data_url(rgb: np.ndarray) -> str:
    """uint8 RGB image -> PNG data URL the browser can put straight in <img src>."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def run_ssr_preview(image_bytes: bytes) -> Tuple[Dict, Dict]:
    """
    Crop + SSR, done up front so the SSR image can stream to the UI before
    classification finishes.

    Returns (previews, state). `previews` goes to the frontend; `state` is
    handed back to build_result() so the pipeline reuses the same crop and
    SSR image instead of recomputing them -- what the user sees is exactly
    what HueView analyzes.
    """
    crop, detection = preprocess_image(image_bytes)
    ssr = compute_ssr(crop)
    previews = {
        "original": _to_data_url(crop),
        "ssr": _to_data_url(ssr),
        "steps": _ssr_steps(crop, ssr),
    }
    return previews, {"crop": crop, "detection": detection, "ssr": ssr}


def _gain_to_rgb(gain: np.ndarray) -> np.ndarray:
    """Gain map as colour, same scheme as ssr_compare.py: white = unchanged,
    blue = darkened, red = brightened, full colour at the gain limits."""
    lo, hi = np.log(ssr_mod.GAIN_LIMITS[0]), np.log(ssr_mod.GAIN_LIMITS[1])
    g = np.log(gain)
    t = np.clip(np.where(g >= 0, g / hi, g / -lo), -1, 1)
    pos, neg = np.clip(t, 0, 1), np.clip(-t, 0, 1)
    vis = np.full(gain.shape + (3,), 255.0)
    vis[..., 0] -= 255 * neg            # less red   -> blue
    vis[..., 1] -= 255 * (pos + neg)
    vis[..., 2] -= 255 * pos            # less blue  -> red
    return np.clip(vis, 0, 255).astype(np.uint8)


def _ssr_steps(crop_rgb: np.ndarray, ssr_rgb: np.ndarray) -> Optional[list]:
    """
    The intermediate images of ssr_normalization.apply_ssr(), for display.

    apply_ssr() only returns its output, so the steps are recomputed here with
    the module's own constants. The output is then rebuilt from these steps
    and must equal the real SSR image pixel for pixel -- if apply_ssr() ever
    changes and this copy drifts, the steps are dropped rather than shown
    wrong. Display only; classification never reads any of this.
    """
    eps = ssr_mod.EPSILON
    img = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR).astype(np.float64) + eps
    lum = 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]
    illum = cv2.GaussianBlur(lum, (0, 0), sigmaX=ssr_mod.SIGMA, sigmaY=ssr_mod.SIGMA)
    retinex = np.log(lum) - np.log(illum)
    gain = (np.exp(retinex) * illum.mean()) / lum
    gain = np.clip(gain ** ssr_mod.STRENGTH, *ssr_mod.GAIN_LIMITS)
    gain *= lum.mean() / (lum * gain).mean()

    rebuilt = np.clip(img * gain[..., None] - eps, 0, 255).astype(np.uint8)
    if not np.array_equal(cv2.cvtColor(rebuilt, cv2.COLOR_BGR2RGB), ssr_rgb):
        log.warning("SSR step preview no longer matches apply_ssr() -- steps omitted.")
        return None

    gray = lambda a: cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    lo, hi = np.percentile(retinex, [1, 99])
    refl = (retinex - lo) / max(hi - lo, 1e-6) * 255     # contrast-stretched for viewing
    sigma = f"{ssr_mod.SIGMA:g}"
    return [
        {"label": "Preprocessed input I", "image": _to_data_url(crop_rgb)},
        {"label": f"Illumination L = Gσ * Y (σ = {sigma})", "image": _to_data_url(gray(illum - eps))},
        {"label": "Reflectance r = log Y − log L", "image": _to_data_url(gray(refl))},
        {"label": f"Gain g ({gain.min():.2f}–{gain.max():.2f}): blue darker, red brighter",
         "image": _to_data_url(_gain_to_rgb(gain))},
        {"label": "SSR output I′ = g · I", "image": _to_data_url(ssr_rgb)},
    ]


def build_result(image_bytes: bytes, filename: str = "", state: Optional[Dict] = None) -> Dict:
    """The full dual-model payload, from the real pipeline."""
    return classify_image(image_bytes, filename, **(state or {}))


def models_loaded() -> Dict[str, bool]:
    return _models_loaded()
