"""
Phase 14.1 -- Frozen training-time artifacts.

Inference-time constants are LOADED, never recomputed. K-Means is not
re-run per image; HSV thresholds are not re-typed. Everything here comes
off disk so the demo can't silently drift from what the study measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..hueview.hsv_skin_filter import FilterConfig, DEFAULT_CONFIG
from .paths import CONFIG_DIR, DATA_DIR

ILLUMINATION_THRESHOLDS_PATH = CONFIG_DIR / "illumination_thresholds.json"
SCC_LABELS_PATH = CONFIG_DIR / "scc_labels.json"
HSV_THRESHOLDS_PATH = CONFIG_DIR / "hsv_skin_thresholds.json"
LEGACY_ILLUMINATION_PATH = DATA_DIR / "illumination_boundaries.json"

#: Same gate as Phase 2's cleaning step. Anything below this was excluded
#: from the study, so the demo must exclude it too.
MIN_FACE_CONFIDENCE = 0.90

#: This project's real Phase 3 k-means output, used only if neither JSON
#: file is present. Not invented -- these are the derived values.
_FALLBACK_ILLUM: Dict = {
    "centroids": [82.19, 126.54, 208.77],
    "b1": 104.37,
    "b2": 167.66,
}

#: Frontend constants.js hardcodes this order, and checkpoints/label_order.json
#: must match it. A mismatch mislabels every prediction without erroring.
DEFAULT_SCC_LABELS = [
    {"id": "SCC-1", "label": "Very Light"},
    {"id": "SCC-2", "label": "Light"},
    {"id": "SCC-3", "label": "Medium"},
    {"id": "SCC-4", "label": "Olive"},
    {"id": "SCC-5", "label": "Brown"},
    {"id": "SCC-6", "label": "Deep"},
]

# --------------------------------------------------------------------------
# OPEN QUESTION -- confirm before trusting HueView's colour numbers
# --------------------------------------------------------------------------
# cielab_features.build_cielab_vector takes an optional original_rgb. When
# passed, CIELAB is sampled from the PRE-SSR crop at the same skin mask
# instead of from SSR pixels. SSR's per-channel rescaling distorts hue
# (hsv_skin_filter.py documents this at length), so the two give materially
# different colour features -- and therefore different Phase 9 undertone
# calls.
#
# Inference must match whatever the Phase 8 batch run used. If it doesn't,
# the demo will disagree with your own results tables, and nothing will
# raise an error to tell you.
#
# CONFIRMED True: run_phase9_batch.py calls build_cielab_vector with
# original_rgb=orig_rgb and documents it as "the evidenced original_rgb
# CIELAB source decided ... from Phase 8.2". qa_phase8_pipeline.py and
# qa_undertone.py do the same. So the study's colour features come from
# the PRE-SSR crop, and inference must too.
CIELAB_FROM_ORIGINAL = True


def load_illumination_thresholds() -> Dict:
    """Phase 3's k-means boundaries. Tries the 14.1 export first, then the
    file Phase 3 already wrote, then the known-good fallback."""
    for path in (ILLUMINATION_THRESHOLDS_PATH, LEGACY_ILLUMINATION_PATH):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return dict(_FALLBACK_ILLUM)


def load_scc_labels() -> list:
    """Readable SCC names, so predictions aren't shown as class indices."""
    if SCC_LABELS_PATH.exists():
        return json.loads(SCC_LABELS_PATH.read_text(encoding="utf-8"))
    return list(DEFAULT_SCC_LABELS)


def load_hsv_config() -> FilterConfig:
    """Phase 7.4's frozen thresholds, including its hsv_source choice, so
    inference filters skin exactly as the batch run did."""
    if HSV_THRESHOLDS_PATH.exists():
        return FilterConfig.from_json(HSV_THRESHOLDS_PATH)
    return DEFAULT_CONFIG


def export_configs() -> None:
    """
    Write the 14.1 artifacts to configs/ if they aren't there yet.

    Run once:  python -m src.inference.config
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not ILLUMINATION_THRESHOLDS_PATH.exists():
        data = load_illumination_thresholds()
        ILLUMINATION_THRESHOLDS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {ILLUMINATION_THRESHOLDS_PATH}: b1={data['b1']}, b2={data['b2']}")
    else:
        print(f"{ILLUMINATION_THRESHOLDS_PATH} already exists -- left alone.")

    if not SCC_LABELS_PATH.exists():
        SCC_LABELS_PATH.write_text(json.dumps(DEFAULT_SCC_LABELS, indent=2), encoding="utf-8")
        print(f"Wrote {SCC_LABELS_PATH}")
    else:
        print(f"{SCC_LABELS_PATH} already exists -- left alone.")

    if not HSV_THRESHOLDS_PATH.exists():
        DEFAULT_CONFIG.to_json(HSV_THRESHOLDS_PATH)
        print(f"Wrote {HSV_THRESHOLDS_PATH} from DEFAULT_CONFIG "
              f"(hsv_source={DEFAULT_CONFIG.hsv_source}) -- "
              f"CHECK this matches your Phase 7.4 batch run.")
    else:
        cfg = load_hsv_config()
        print(f"{HSV_THRESHOLDS_PATH} already exists (hsv_source={cfg.hsv_source}) -- left alone.")


# Loaded once at import.
ILLUM = load_illumination_thresholds()
SCC_LABELS = load_scc_labels()
HSV_CONFIG = load_hsv_config()


if __name__ == "__main__":
    export_configs()
