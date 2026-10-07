"""
Phase 14.6 -- run_hueview()

SSR -> MediaPipe landmarks -> regional segmentation -> HSV skin filtering ->
CIELAB -> undertone (manuscript path, for the undertone + L*a*b* display).

CLASSIFICATION (SCC) feeds each head exactly what train.py v3 trained it on,
built from the same Phase 7.4 patches computed below: regional heads get the
SSR patch with non-skin pixels zeroed plus a 3-D skimage CIELAB skin mean;
full_face gets the unmasked SSR face plus the 15-D regional CIELAB vector.
CIELAB comes from train.py's own cielab_mean(), so the values are identical.

Ordering note: landmarks from the ORIGINAL crop; masking on the SSR image.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from ..hueview.segment_regions import REGIONS, build_region_mask
from ..hueview.ssr_normalization import apply_ssr
from ..hueview.hsv_skin_filter import filter_all_regions
from ..hueview.region_selector import RegionalConfigurationSelector, FULL_FACE
from ..hueview.cielab_features import build_cielab_vector
from ..hueview.regions import REGION_ORDER, REGION_DISPLAY
from ..hueview.undertone import compute_undertone_descriptor
from ..hueview.train import cielab_mean
from .config import HSV_CONFIG, CIELAB_FROM_ORIGINAL
from .models import load_models
from .paths import find_landmarker
from .preprocess import NoFaceDetected

LANDMARKER_MODEL = find_landmarker()

SCC_CLASS_ORDER = ["SCC-1", "SCC-2", "SCC-3", "SCC-4", "SCC-5", "SCC-6"]
_NULL_CLS = {"scc": None, "probabilities": None, "confidence": None, "margin": None}

_SELECTOR = RegionalConfigurationSelector()
_landmarker = None


def compute_ssr(crop_rgb: np.ndarray) -> np.ndarray:
    """apply_ssr() on an RGB crop, returned as RGB.

    apply_ssr weights luminance for BGR input, because the batch run feeds it
    cv2.imread output. The upload crop is RGB (PIL), so swap in and out --
    otherwise R and B trade luminance weights and the gain map no longer
    matches the one the batch run produced.
    """
    import cv2
    ssr_bgr = apply_ssr(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(ssr_bgr, cv2.COLOR_BGR2RGB)


def _get_landmarker():
    """Built once -- MediaPipe has real startup cost (dev plan 14.2)."""
    global _landmarker
    if _landmarker is None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise RuntimeError("mediapipe is not installed. pip install mediapipe") from e

        if not LANDMARKER_MODEL.exists():
            raise RuntimeError(
                f"{LANDMARKER_MODEL} not found. Copy it in from Phase 7.1 -- "
                f"inference must use the same landmarker the batch run used."
            )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_MODEL)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def extract_landmarks(crop_rgb: np.ndarray) -> Optional[np.ndarray]:
    import mediapipe as mp

    landmarker = _get_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_rgb))
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    h, w = crop_rgb.shape[:2]
    coords = np.array(
        [[lm.x * w, lm.y * h, lm.z] for lm in result.face_landmarks[0]],
        dtype=np.float32,
    )
    return coords[:468]


# ---------------- classification helpers (train.py v3 contract) ------------
def _lab_input(region: str, patches, scaler) -> np.ndarray:
    """train.py's lab_vector(), on the patches this run just computed.

    Single region -> 3-D skin mean; full_face -> 15-D, REGION_ORDER, gaps
    filled with the face's own mean. A vector with no usable region falls
    back to scaler.mean_, which equals train.py's train-split fallback: the
    scaler was fit on rows already imputed with that mean.
    """
    if region != FULL_FACE:
        p = patches[region]
        v = cielab_mean(p.image, p.skin_mask)
        return scaler.mean_.astype(np.float32) if v is None else v

    parts = [cielab_mean(patches[r].image, patches[r].skin_mask) for r in REGION_ORDER]
    flags = np.array([v is not None for v in parts])
    if not flags.any():
        return scaler.mean_.astype(np.float32)
    parts = np.stack([np.full(3, np.nan, np.float32) if v is None else v for v in parts])
    parts[~flags] = parts[flags].mean(axis=0)
    return parts.reshape(-1).astype(np.float32)


def _img_input(region: str, ssr: np.ndarray, patches) -> np.ndarray:
    """train.py's HueViewSeq image: full_face gets the unmasked SSR face;
    each region gets its SSR patch with every non-skin pixel zeroed."""
    if region == FULL_FACE:
        return ssr
    p = patches[region]
    return p.image * p.skin_mask[..., None]


def _predict_one(model, scaler, img_patch: np.ndarray, lab_raw: np.ndarray) -> Dict:
    img_in = np.expand_dims(img_patch.astype("float32"), 0)          # 0-255, EffNet rescales
    lab_scaled = scaler.transform(lab_raw.reshape(1, -1)).astype("float32")
    probs = model.predict({"img": img_in, "lab": lab_scaled}, verbose=0)[0]
    order = np.sort(probs)[::-1]
    idx = int(np.argmax(probs))
    return {
        "scc": SCC_CLASS_ORDER[idx],
        "probabilities": probs.tolist(),
        "confidence": float(probs[idx]),
        "margin": float(order[0] - order[1]),
    }


def _classify_hueview(ssr, patches, hv) -> Dict:
    models, scalers = hv["models"], hv["scalers"]
    out = {"regions": {}, "full_face": None, "headline": None}

    all_probs = []
    for name in REGION_ORDER:
        res = _predict_one(models[name], scalers[name],
                           _img_input(name, ssr, patches),
                           _lab_input(name, patches, scalers[name]))
        out["regions"][name] = res
        all_probs.append(np.array(res["probabilities"], dtype=np.float64))

    ff_res = _predict_one(models[FULL_FACE], scalers[FULL_FACE],
                          _img_input(FULL_FACE, ssr, patches),
                          _lab_input(FULL_FACE, patches, scalers[FULL_FACE]))
    out["full_face"] = ff_res
    all_probs.append(np.array(ff_res["probabilities"], dtype=np.float64))  # full_face isasama na sa headline

    mean = np.mean(all_probs, axis=0)
    order = np.sort(mean)[::-1]
    idx = int(np.argmax(mean))
    out["headline"] = {
        "scc": SCC_CLASS_ORDER[idx],
        "probabilities": mean.tolist(),
        "confidence": float(mean[idx]),
        "margin": float(order[0] - order[1]),
    }
    return out


def run_hueview(crop_rgb: np.ndarray, ssr: Optional[np.ndarray] = None) -> Dict:
    """`ssr` lets the API pass in the SSR image it already computed for the
    preview, so the user sees exactly the image this run analyzes."""
    t0 = time.perf_counter()

    if ssr is None:
        ssr = compute_ssr(crop_rgb)

    landmarks = extract_landmarks(crop_rgb)
    if landmarks is None:
        raise NoFaceDetected(
            "MTCNN found a face but MediaPipe could not place landmarks on it. "
            "Regional analysis needs the mesh, so HueView can't run on this image."
        )

    masks = {
        name: build_region_mask(landmarks, idxs, ssr.shape).astype(bool)
        for name, idxs in REGIONS.items()
    }

    patches = filter_all_regions(
        ssr_image=ssr,
        geometric_masks=masks,
        config=HSV_CONFIG,
        reference_image=crop_rgb,
    )

    configs = _SELECTOR.route_all(patches, image_id="inference")
    full = configs.get(FULL_FACE)

    # ---- SCC classification (train.py v3: HSV skin masks + skimage CIELAB) ----
    hv = load_models()["hueview"]
    is_placeholder = hv is None
    cls = _classify_hueview(ssr, patches, hv) if not is_placeholder else None

    original = crop_rgb if CIELAB_FROM_ORIGINAL else None
    region_rows = []
    face_lab = {"L": None, "a": None, "b": None}
    undertone_block = {
        "method": "hue_angle",
        "label": None,
        "hue_angle_deg": None,
        "source": "Mean of the selected regional CIELAB values",
        "distribution": None,
        "was_tied": None,
    }

    if full is not None:
        vector = build_cielab_vector(full, original_rgb=original)
        descriptor = compute_undertone_descriptor(full, vector)

        per_region = vector.reshape(-1, 3)
        imputed = set(full.imputed_regions)

        for name, (L, a, b) in zip(full.component_order(), per_region):
            patch = patches.get(name)
            info = descriptor["regions"].get(name, {})
            cls_r = cls["regions"][name] if cls else _NULL_CLS
            region_rows.append({
                "name": REGION_DISPLAY[name],
                "region_key": name,
                "L": round(float(L), 2),
                "a": round(float(a), 2),
                "b": round(float(b), 2),
                "pixels": int(patch.n_valid_px) if patch else 0,
                "used": name not in imputed,
                "imputed": name in imputed,
                "status": patch.status if patch else "empty",
                "retention": round(patch.retention, 3) if patch else 0.0,
                "hue_angle_deg": round(float(info.get("hue_deg")), 2) if info.get("hue_deg") is not None else None,
                "undertone": info.get("undertone"),
                **cls_r,
            })

        face_mean = per_region.mean(axis=0)
        face_lab = {
            "L": round(float(face_mean[0]), 2),
            "a": round(float(face_mean[1]), 2),
            "b": round(float(face_mean[2]), 2),
        }

        hues = [r["hue_angle_deg"] for r in region_rows if r["hue_angle_deg"] is not None]
        labels = [r["undertone"] for r in region_rows if r["undertone"]]
        n = len(labels) or 1
        undertone_block = {
            "method": "hue_angle",
            "label": descriptor["majority_undertone"],
            "hue_angle_deg": round(float(np.mean(hues)), 2) if hues else None,
            "source": "Mean of the selected regional CIELAB values",
            "distribution": {
                "warm": round(labels.count("Warm") / n, 3),
                "neutral": round(labels.count("Neutral") / n, 3),
                "cool": round(labels.count("Cool") / n, 3),
            },
            "was_tied": descriptor["was_tied"],
        }

        full_hue = float(np.degrees(np.arctan2(face_mean[2], face_mean[1])) % 360)
        ff_cls = cls["full_face"] if cls else _NULL_CLS
        region_rows.append({
            "name": REGION_DISPLAY[FULL_FACE],
            "region_key": FULL_FACE,
            **face_lab,
            "pixels": int(full.n_valid_px),
            "used": False,
            "imputed": False,
            "status": full.status,
            "retention": None,
            "hue_angle_deg": round(full_hue, 2),
            "undertone": None,
            **ff_cls,
        })
    else:
        for name in REGION_ORDER:
            patch = patches.get(name)
            cls_r = cls["regions"][name] if cls else _NULL_CLS
            region_rows.append({
                "name": REGION_DISPLAY[name],
                "region_key": name,
                "L": None, "a": None, "b": None,
                "pixels": int(patch.n_valid_px) if patch else 0,
                "used": False, "imputed": False,
                "status": patch.status if patch else "empty",
                "retention": round(patch.retention, 3) if patch else 0.0,
                "hue_angle_deg": None, "undertone": None,
                **cls_r,
            })

    # ---- headline SCC = ensemble mean of all 6 region softmaxes ----
    if cls:
        head = cls["headline"]
        scc, probabilities = head["scc"], head["probabilities"]
        confidence, margin = head["confidence"], head["margin"]
    else:
        scc = probabilities = confidence = margin = None

    return {
        "name": "HueView",
        "method": "SSR + regional segmentation",
        "checkpoint": "no weights loaded" if is_placeholder else "loaded",
        "placeholder": is_placeholder,
        "scc": scc,
        "probabilities": probabilities,
        "confidence": confidence,
        "margin": margin,
        "lab": face_lab,
        "regions": region_rows,
        "undertone": undertone_block,
        "inference_ms": int((time.perf_counter() - t0) * 1000),
    }