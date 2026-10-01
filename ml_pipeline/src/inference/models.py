"""
Phase 14.2 -- Model loading (once, at startup)
================================================
Baseline: final weights (manuscript-compliant).
HueView: PROVISIONAL per-region weights -- trained geometric-mask-only, NOT the
manuscript HSV-filtered way. FOR DEMO ONLY. Will be replaced with final weights 
after manuscript retrain.
"""

from __future__ import annotations

import logging

from tensorflow import keras

from .paths import MODELS_DIR

logger = logging.getLogger(__name__)

BASELINE_WEIGHTS_FILENAME = "baseline_effnet.keras"

HUEVIEW_REGIONS = ["forehead", "left_cheek", "right_cheek", "nose_bridge", "jawline", "full_face"]
HUEVIEW_SUFFIX = "_final_candidate"   # <-- swap to "" after manuscript retrain

_models = {"baseline": None, "hueview": None}
_loaded = False


def load_models() -> dict:
    global _loaded
    if _loaded:
        return _models

    baseline_path = MODELS_DIR / BASELINE_WEIGHTS_FILENAME
    if baseline_path.exists():
        logger.info("Loading baseline model from %s", baseline_path)
        _models["baseline"] = keras.models.load_model(baseline_path, compile=False)
        logger.info("Baseline model loaded.")
    else:
        logger.warning("Baseline weights not found at %s -- placeholder mode.", baseline_path)
        _models["baseline"] = None

    _models["hueview"] = _load_hueview()

    _loaded = True
    return _models


def _load_hueview():
    try:
        import joblib
    except ImportError:
        logger.warning("joblib not installed -- HueView placeholder mode. pip install scikit-learn")
        return None

    models, scalers, missing = {}, {}, []
    for region in HUEVIEW_REGIONS:
        mpath = MODELS_DIR / f"hueview_{region}{HUEVIEW_SUFFIX}.h5"
        spath = MODELS_DIR / f"lab_scaler_{region}{HUEVIEW_SUFFIX}.pkl"
        if not mpath.exists():
            missing.append(mpath.name); continue
        if not spath.exists():
            missing.append(spath.name); continue
        models[region] = keras.models.load_model(mpath, compile=False)
        scalers[region] = joblib.load(spath)

    if missing:
        logger.warning("HueView placeholder mode -- missing %d file(s): %s",
                       len(missing), ", ".join(missing))
        return None

    logger.info("HueView loaded (%d regions, PROVISIONAL geometric-trained weights).", len(models))
    return {"models": models, "scalers": scalers}


def models_loaded() -> dict:
    return {
        "baseline": _models["baseline"] is not None,
        "hueview": _models["hueview"] is not None,
    }