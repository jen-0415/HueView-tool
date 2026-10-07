"""
Phase 15 -- inference wrapper.

The bridge between the API layer and Phase 14. routes.py imports from here
and never touches the pipeline directly, so the web service and the model
code stay independent of each other.

run_detect() and build_result() call the real pipeline. If a model's
weights are missing, that model's SCC fields come back null and its
`placeholder` flag is true -- the frontend renders that state.
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
from ..inference.hueview import compute_ssr, _img_input
from ..inference.models import models_loaded as _models_loaded
from ..hueview import ssr_normalization as ssr_mod
from ..hueview.regions import REGION_DISPLAY, REGION_ORDER

log = logging.getLogger(__name__)

# Stage keys streamed as SSE "stage" events, in this order. routes.py runs
# "detect" and "ssr" for real and treats the rest as progress beats.
STAGE_KEYS = ["detect", "ssr", "segment", "cnn", "lab", "undertone"]

# Must match frontend/src/constants.js, configs/scc_labels.json and the
# models' output order, or every prediction is mislabeled without erroring.
SCC_LABELS = ["SCC-1", "SCC-2", "SCC-3", "SCC-4", "SCC-5", "SCC-6"]

REGION_NAMES = [REGION_DISPLAY[r] for r in REGION_ORDER] + [REGION_DISPLAY["full_face"]]


def run_detect(image_bytes: bytes, filename: str = "") -> Dict:
    """
    POST /api/detect -- the fast face check before committing to full
    inference. Real MTCNN now, with Phase 2's exact crop and gates.
    """
    crop, detection = preprocess_image(image_bytes)
    return {
        "confidence": detection["confidence"],
        "landmarks_found": detection["landmarks_found"],
        "bbox": detection["bbox"],
        "upscaled": detection["upscaled"],
        "crop_preview": _to_data_url(crop),   # the exact 224x224 crop both models get
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
    """The full dual-model payload, from the real pipeline, plus the HueView
    segmentation visuals and a description of how its final SCC is chosen."""
    capture: Dict = {}
    payload = classify_image(image_bytes, filename, **(state or {}), capture=capture)

    hueview = payload["models"]["hueview"]
    hueview["decision"] = {
        "method": "majority_vote",
        "heads": list(REGION_ORDER) + ["full_face"],
        "description": (
            "Each of the six HueView models (forehead, left cheek, right cheek, "
            "nose bridge, jawline and full face) votes for the SCC class it rates "
            "highest. The class with the most votes is the final SCC. If classes "
            "tie on votes, the tied class with the highest mean probability across "
            "the six models wins."
        ),
    }
    try:
        crop = (state or {}).get("crop")
        if crop is None:
            crop, _ = preprocess_image(image_bytes)
        payload["image"]["preview"] = _to_data_url(crop)   # Results shows the real crop
        hueview["segmentation"] = _segmentation_visuals(crop, capture)
    except Exception:
        # Display extra only -- never fail an analysis over a picture.
        log.exception("Segmentation visuals failed")
        hueview["segmentation"] = None
    return payload


# Display colours per region (RGB), also sent to the UI for its legend.
REGION_COLORS = {
    "forehead": (245, 158, 66),
    "left_cheek": (52, 168, 140),
    "right_cheek": (88, 120, 220),
    "nose_bridge": (214, 72, 160),
    "jawline": (120, 180, 40),
}


def _overlay(base: np.ndarray, masks: Dict[str, np.ndarray], alpha: float = 0.5,
             outlines: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    out = base.astype(np.float64)
    for name, m in masks.items():
        out[m] = (1 - alpha) * out[m] + alpha * np.array(REGION_COLORS[name])
    out = np.clip(out, 0, 255).astype(np.uint8)
    for name, m in (outlines or {}).items():
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, REGION_COLORS[name], 1)
    return out


def _zoom_to(img: np.ndarray, mask: np.ndarray, size: int = 160, pad: int = 6) -> np.ndarray:
    """Crop to the mask's bounding box (letterboxed) so a small region is legible."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((size, size, 3), np.uint8)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
    sub = img[y0:y1, x0:x1]
    s = size / max(sub.shape[:2])
    sub = cv2.resize(sub, (max(1, int(sub.shape[1] * s)), max(1, int(sub.shape[0] * s))),
                     interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - sub.shape[0]) // 2, (size - sub.shape[1]) // 2
    canvas[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
    return canvas


def _segmentation_visuals(crop: np.ndarray, capture: Dict) -> Dict:
    """
    Pictures of what HueView's classification actually used, built from the
    arrays run_hueview() captured -- nothing is recomputed:
      * geometric: Phase 7.3 landmark convex-hull regions on the crop
      * skin:      pixels kept by Phase 7.4's HSV filter (removed ones greyed)
      * inputs:    each model's exact image input (skin-masked SSR patch,
                   zoomed to its region; unmasked SSR face for full_face)
    """
    ssr, masks, patches = capture["ssr"], capture["masks"], capture["patches"]
    skin = {n: patches[n].skin_mask for n in REGION_ORDER}

    any_geom = np.zeros(crop.shape[:2], bool)
    for n in REGION_ORDER:
        any_geom |= masks[n]
    any_skin = np.zeros(crop.shape[:2], bool)
    for m in skin.values():
        any_skin |= m
    removed = crop.copy()
    removed[any_geom & ~any_skin] = (removed[any_geom & ~any_skin] * 0.25).astype(np.uint8)

    inputs = {n: _zoom_to(_img_input(n, ssr, patches), masks[n]) for n in REGION_ORDER}
    inputs["full_face"] = cv2.resize(_img_input("full_face", ssr, patches), (160, 160),
                                     interpolation=cv2.INTER_AREA)
    return {
        "geometric": _to_data_url(_overlay(crop, masks, 0.45, outlines=masks)),
        "skin": _to_data_url(_overlay(removed, skin, 0.55, outlines=masks)),
        "colors": {n: "#%02x%02x%02x" % c for n, c in REGION_COLORS.items()},
        "inputs": {n: _to_data_url(img) for n, img in inputs.items()},
        "geometric_pixels": {n: int(masks[n].sum()) for n in REGION_ORDER},
    }


def models_loaded() -> Dict[str, bool]:
    return _models_loaded()
