"""
Phase 14.2 -- Model loading.

PLACEHOLDER: no .h5/.keras weights exist in this repo yet. Weight files are
gitignored, so they have to come from whoever trained them, via Drive --
the same way landmarks.npy did.

Everything else in Phase 14 produces real numbers already. This module is
the only thing standing between the demo and real SCC predictions.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..hueview.region_selector import CONFIGURATIONS
from .paths import MODELS_DIR

_MODELS: Dict[str, Optional[object]] = {"baseline": None, "hueview": None}


def load_models() -> Dict[str, Optional[object]]:
    """
    Called ONCE at API startup (never per request).

    When weights arrive, this becomes roughly:

        from tensorflow.keras.models import load_model

        _MODELS["baseline"] = load_model(f"{MODELS_DIR}/baseline_effnet.h5")

        # Phase 8.1's shared-vs-per-region decision determines which of these
        # two is right -- check what Phase 10 actually saved before wiring it.
        _MODELS["hueview"] = {
            cfg: load_model(f"{MODELS_DIR}/hueview_{cfg}.h5")
            for cfg in CONFIGURATIONS
        }

    Returns the dict either way, so callers don't need to care.
    """
    return _MODELS


def models_loaded() -> Dict[str, bool]:
    """What /api/health reports."""
    return {name: obj is not None for name, obj in _MODELS.items()}


def any_loaded() -> bool:
    return any(models_loaded().values())
