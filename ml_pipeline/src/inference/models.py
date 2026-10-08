"""
Phase 14.2 -- Model loading (once, at startup)
================================================
Baseline: baseline_effnet_final.h5.
HueView: per-configuration weights from train.py (manuscript-aligned:
HSV-filtered skin masks, skimage CIELAB), the "_final1" files. SIX
configurations load -- the five facial regions plus Full Face, a separately
trained whole-face classifier. All six vote on the final SCC
(inference/hueview.py).
"""

from __future__ import annotations

import logging

from tensorflow import keras

from .paths import MODELS_DIR

logger = logging.getLogger(__name__)

BASELINE_WEIGHTS_FILENAME = "baseline_effnet_final.h5"

# The six configurations, in the order Figure 11 / Table 28 list them.
# "full_face" is a configuration, not a sixth anatomical region: its CNN takes
# the whole SSR face and its CIELAB input is the 15-value regional profile.
HUEVIEW_REGIONS = ["forehead", "left_cheek", "right_cheek", "nose_bridge",
                   "jawline", "full_face"]
# train.py run_tag suffixes, in order of preference. Each configuration loads
# the first suffix whose model AND scaler both exist.
#
# "_final1" only -- the current weights for all six configurations, regional
# and full_face alike. Pinned to a single tag on purpose: with a fallback a
# missing file would quietly serve a *different* set of weights, which is how
# it stopped being obvious which models the app was actually running. If one
# is missing the app starts in placeholder mode and logs which, rather than
# mixing tags. ("_final_candidate" predates the current input contract.)
#
# Keep in step with evaluate.py --run-tag, or the app and the thesis tables
# will report different predictions; results/model_provenance.csv records the
# tag the last evaluation run actually scored.
HUEVIEW_SUFFIXES = ("_final1",)

_models = {"baseline": None, "hueview": None}
_files = {"baseline": None, "hueview": {}}   # what actually got loaded
_loaded = False


def load_models() -> dict:
    global _loaded
    if _loaded:
        return _models

    baseline_path = MODELS_DIR / BASELINE_WEIGHTS_FILENAME
    if baseline_path.exists():
        logger.info("Loading baseline model from %s", baseline_path)
        _models["baseline"] = keras.models.load_model(baseline_path, compile=False)
        _files["baseline"] = baseline_path.name
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

    models, scalers, chosen, missing = {}, {}, {}, []
    for region in HUEVIEW_REGIONS:
        for suffix in HUEVIEW_SUFFIXES:
            mpath = MODELS_DIR / f"hueview_{region}{suffix}.h5"
            spath = MODELS_DIR / f"lab_scaler_{region}{suffix}.pkl"
            if mpath.exists() and spath.exists():
                models[region] = keras.models.load_model(mpath, compile=False)
                scalers[region] = joblib.load(spath)
                chosen[region] = suffix
                logger.info("HueView %-12s <- %s + %s", region, mpath.name, spath.name)
                break
        else:
            missing.append(region)

    if missing:
        logger.warning("HueView placeholder mode -- no model+scaler pair for %s (tried suffixes %s)",
                       ", ".join(missing), ", ".join(HUEVIEW_SUFFIXES))
        return None

    _files["hueview"] = chosen
    return {"models": models, "scalers": scalers}


def model_files() -> dict:
    """Which weights were loaded: {"baseline": filename, "hueview": {region: suffix}}."""
    return {"baseline": _files["baseline"], "hueview": dict(_files["hueview"])}


def checkpoint_label(model: str) -> str:
    """Short description of the loaded weights for the UI's "ckpt" field."""
    if model == "baseline":
        return _files["baseline"] or "no weights loaded"
    by_suffix: dict = {}
    for region, suffix in _files["hueview"].items():
        by_suffix.setdefault(suffix.lstrip("_"), []).append(region)
    if not by_suffix:
        return "no weights loaded"
    return "; ".join(f"{tag}: {', '.join(regions)}" if len(by_suffix) > 1 else tag
                     for tag, regions in by_suffix.items())


def models_loaded() -> dict:
    return {
        "baseline": _models["baseline"] is not None,
        "hueview": _models["hueview"] is not None,
    }