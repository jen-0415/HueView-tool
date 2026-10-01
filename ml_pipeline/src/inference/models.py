"""
Phase 14.2 -- Model loading (once, at startup)
================================================
Loads the Baseline and HueView CNN models a single time. Baseline weights
exist now (Stage 2, 2026-09-06 run -- val_loss 0.938, val_accuracy 62.44%,
beat Stage 1's 0.9713/59.94%, so no fallback checkpoint needed). HueView
weights don't exist yet, so it stays in placeholder mode until Phase 10/8.1
finishes -- run_hueview() already handles model=None the same way
run_baseline() did before this file was wired up.
"""

from __future__ import annotations

import logging

from tensorflow import keras

from .paths import MODELS_DIR

logger = logging.getLogger(__name__)

BASELINE_WEIGHTS_FILENAME = "baseline_effnet.keras"
HUEVIEW_WEIGHTS_FILENAME = "hueview_effnet.keras"

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
        logger.info(
            "Baseline model loaded. Inputs: %s Output: %s",
            [i.name for i in _models["baseline"].inputs],
            _models["baseline"].output_shape,
        )
    else:
        logger.warning(
            "Baseline weights not found at %s -- run_baseline() stays in "
            "placeholder mode until it's downloaded there.", baseline_path,
        )
        _models["baseline"] = None

    hueview_path = MODELS_DIR / HUEVIEW_WEIGHTS_FILENAME
    if hueview_path.exists():
        logger.info("Loading HueView model from %s", hueview_path)
        _models["hueview"] = keras.models.load_model(hueview_path, compile=False)
        logger.info("HueView model loaded.")
    else:
        logger.info(
            "HueView weights not found at %s -- expected for now (training "
            "not finished). run_hueview() stays in placeholder mode.",
            hueview_path,
        )
        _models["hueview"] = None

    _loaded = True
    return _models


def models_loaded() -> dict:
    return {
        "baseline": _models["baseline"] is not None,
        "hueview": _models["hueview"] is not None,
    }