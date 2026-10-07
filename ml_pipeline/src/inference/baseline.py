"""
Phase 14.5 -- run_baseline()

Global RGB averaging + the rule-based undertone branch. Both are REAL --
neither needs trained weights. SCC classification is now real too, once
baseline_effnet.keras is present; falls back to placeholder if it isn't.

Reuses src/baseline/undertone.py directly rather than reimplementing the
thresholds, so the demo can't drift from what Phase 6.3 measured.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from ..baseline.undertone import classify_undertone, undertone_ratios
from .models import load_models

# Fixed class order from Stage 2 training (baseline_training_summary.json
# "classes" field). idx 0 -> SCC-1 ... idx 5 -> SCC-6.
SCC_CLASS_ORDER = ["SCC-1", "SCC-2", "SCC-3", "SCC-4", "SCC-5", "SCC-6"]


def run_baseline(crop_rgb: np.ndarray) -> Dict:
    t0 = time.perf_counter()

    # 6.1 -- per-channel mean over the whole crop, 0-255.
    # undertone.py is scale-invariant, so no need to divide by 255 here;
    # the CNN branch needs the normalized version, handled below.
    means = crop_rgb.reshape(-1, 3).mean(axis=0)
    r_ratio, g_ratio, b_ratio = undertone_ratios(means)
    label = classify_undertone(means)

    # ---------------- CLASSIFY (real, now that weights exist) -------------
    model = load_models()["baseline"]
    is_placeholder = model is None
    scc = probabilities = confidence = margin = None

    if not is_placeholder:
        cnn_in = np.expand_dims(crop_rgb.astype("float32"), 0)   # un-normalized
        rgb_in = np.expand_dims(means / 255.0, 0)
        probs = model.predict([cnn_in, rgb_in], verbose=0)[0]
        idx = int(probs.argmax())
        scc = SCC_CLASS_ORDER[idx]
        probabilities = probs.tolist()
        confidence = float(probs[idx])
        margin = confidence - float(np.sort(probs)[-2])
    # ----------------------------------------------------------------------

    return {
        "name": "Baseline",
        "method": "EfficientNetB0 + global RGB mean",
        "checkpoint": "no weights loaded" if is_placeholder else "loaded",
        "placeholder": is_placeholder,
        "scc": scc,
        "probabilities": probabilities,
        "confidence": confidence,
        "margin": margin,
        "rgb": {
            "R": int(round(float(means[0]))),
            "G": int(round(float(means[1]))),
            "B": int(round(float(means[2]))),
        },
        "regions": None,  # baseline is global by definition
        "undertone": {
            "method": "rgb_ratio",
            "label": label,
            "ratios": {"r": round(r_ratio, 4), "g": round(g_ratio, 4), "b": round(b_ratio, 4)},
            "b_ratio": round(b_ratio, 4),
            "source": "Global RGB mean",
            "distribution": None,  # single global value, no per-region spread
        },
        "inference_ms": int((time.perf_counter() - t0) * 1000),
    }