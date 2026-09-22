"""
Phase 14.6 -- run_hueview()

The full HueView path for one image: SSR -> MediaPipe landmarks ->
regional segmentation -> HSV skin filtering -> CIELAB -> undertone.

Everything here is REAL except the SCC classification. Each stage calls the
same module the batch runs used, so a number shown in the demo is produced
by the same code that produced the results tables.

Ordering note, per the manuscript: landmarks come from the ORIGINAL crop
(SSR strips the contrast MediaPipe needs), masking is applied to the
SSR-NORMALIZED image.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..hueview.segment_regions import REGIONS, build_region_mask
from ..hueview.ssr_normalization import apply_ssr
from ..hueview.hsv_skin_filter import filter_all_regions
from ..hueview.region_selector import RegionalConfigurationSelector, FULL_FACE
from ..hueview.cielab_features import build_cielab_vector
from ..hueview.regions import REGION_ORDER, REGION_DISPLAY
from ..hueview.undertone import compute_undertone_descriptor
from .config import HSV_CONFIG, CIELAB_FROM_ORIGINAL
from .models import load_models
from .paths import find_landmarker
from .preprocess import NoFaceDetected

LANDMARKER_MODEL = find_landmarker()

_SELECTOR = RegionalConfigurationSelector()
_landmarker = None


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
            min_face_detection_confidence=0.5,  # matches landmark_extraction.py
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def extract_landmarks(crop_rgb: np.ndarray) -> Optional[np.ndarray]:
    """468 landmarks in pixel coords -- same format as Phase 7.1's
    landmarks.npy rows, so build_region_mask consumes them unchanged."""
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
    return coords[:468]  # drop the 10 iris points, matching Phase 7.1


def run_hueview(crop_rgb: np.ndarray) -> Dict:
    t0 = time.perf_counter()

    # --- 7.2 SSR (real) ---------------------------------------------------
    ssr = apply_ssr(crop_rgb)

    # --- 7.1 landmarks, on the ORIGINAL crop (real) -----------------------
    landmarks = extract_landmarks(crop_rgb)
    if landmarks is None:
        raise NoFaceDetected(
            "MTCNN found a face but MediaPipe could not place landmarks on it. "
            "Regional analysis needs the mesh, so HueView can't run on this image."
        )

    # --- 7.3 geometric masks (real) ---------------------------------------
    masks = {
        name: build_region_mask(landmarks, idxs, ssr.shape).astype(bool)
        for name, idxs in REGIONS.items()
    }

    # --- 7.4 HSV skin filtering, frozen config (real) ---------------------
    patches = filter_all_regions(
        ssr_image=ssr,
        geometric_masks=masks,
        config=HSV_CONFIG,
        reference_image=crop_rgb,  # only used when hsv_source == "original"
    )

    # --- 7.5 routing (real) -----------------------------------------------
    configs = _SELECTOR.route_all(patches, image_id="inference")
    full = configs.get(FULL_FACE)

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
        # --- 8.2 CIELAB (real) -- one 15-d vector, 3 per region in
        # REGION_ORDER. Taking per-region values from this single vector
        # (rather than recomputing each region separately) guarantees the
        # displayed L*a*b* are exactly what the undertone vote used.
        vector = build_cielab_vector(full, original_rgb=original)

        # --- 9 undertone (real) -- their module, including the 340-degree
        # hue-wraparound correction and the 60-degree tie-break.
        descriptor = compute_undertone_descriptor(full, vector)

        per_region = vector.reshape(-1, 3)
        imputed = set(full.imputed_regions)

        for name, (L, a, b) in zip(full.component_order(), per_region):
            patch = patches.get(name)
            info = descriptor["regions"].get(name, {})
            region_rows.append({
                "name": REGION_DISPLAY[name],
                "region_key": name,
                "L": round(float(L), 2),
                "a": round(float(a), 2),
                "b": round(float(b), 2),
                "pixels": int(patch.n_valid_px) if patch else 0,
                # Imputed regions had no usable skin -- their values are
                # borrowed from the rest of the face, so flag them as not used.
                "used": name not in imputed,
                "imputed": name in imputed,
                "status": patch.status if patch else "empty",
                "retention": round(patch.retention, 3) if patch else 0.0,
                "hue_angle_deg": round(float(info.get("hue_deg")), 2) if info.get("hue_deg") is not None else None,
                "undertone": info.get("undertone"),
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

        # Full Face row, for the regions table in the UI.
        full_hue = float(np.degrees(np.arctan2(face_mean[2], face_mean[1])) % 360)
        region_rows.append({
            "name": REGION_DISPLAY[FULL_FACE],
            "region_key": FULL_FACE,
            **face_lab,
            "pixels": int(full.n_valid_px),
            "used": False,  # composite, not a voter
            "imputed": False,
            "status": full.status,
            "retention": None,
            "hue_angle_deg": round(full_hue, 2),
            "undertone": None,
        })
    else:
        # Every region failed -- report the empties rather than inventing values.
        for name in REGION_ORDER:
            patch = patches.get(name)
            region_rows.append({
                "name": REGION_DISPLAY[name],
                "region_key": name,
                "L": None, "a": None, "b": None,
                "pixels": int(patch.n_valid_px) if patch else 0,
                "used": False, "imputed": False,
                "status": patch.status if patch else "empty",
                "retention": round(patch.retention, 3) if patch else 0.0,
                "hue_angle_deg": None, "undertone": None,
            })

    # ---------------- CLASSIFY (placeholder -- needs weights) -------------
    # Real version, per Phase 8.3:
    #   models = load_models()["hueview"]
    #   model = models[chosen_config]           # or the shared model
    #   cnn_in = np.expand_dims(cfg.image.astype("float32"), 0)
    #   lab_in = np.expand_dims(vector, 0)       # 3-d or 15-d
    #   probs = model.predict([cnn_in, lab_in])[0]
    models = load_models()["hueview"]
    is_placeholder = models is None
    scc = probabilities = confidence = margin = None
    # ----------------------------------------------------------------------

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
