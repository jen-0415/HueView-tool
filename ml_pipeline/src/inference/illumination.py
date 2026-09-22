"""
Phase 14.4 -- classify_illumination()

Mean Y in YCrCb, binned against Phase 3's frozen k-means boundaries.
Shared by both pipelines, so it's computed once per image and passed in.

The boundaries are LOADED, not recomputed -- k-means never runs at
inference time. Re-fitting on a single image would be meaningless.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .config import ILLUM, _FALLBACK_ILLUM


def classify_illumination(crop_rgb: np.ndarray) -> Dict:
    ycrcb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb)
    mean_y = float(ycrcb[:, :, 0].mean())

    b1 = float(ILLUM["b1"])
    b2 = float(ILLUM["b2"])
    centroids = ILLUM.get("centroids", _FALLBACK_ILLUM["centroids"])

    # Boundaries are inclusive on the Medium side, matching Phase 3's
    # assign_labels.py: Low if < b1, Medium if b1 <= y <= b2, else High.
    if mean_y < b1:
        name, index = "Low", 1
    elif mean_y <= b2:
        name, index = "Medium", 2
    else:
        name, index = "High", 3

    return {
        "metric": "Mean luminance",
        "value": round(mean_y, 2),
        "scale": [0, 255],
        "bin": name,
        "bin_index": index,
        "bins": [
            {"name": "Low", "centroid": round(float(centroids[0]), 2),
             "range": [0, round(b1, 2)]},
            {"name": "Medium", "centroid": round(float(centroids[1]), 2),
             "range": [round(b1, 2), round(b2, 2)]},
            {"name": "High", "centroid": round(float(centroids[2]), 2),
             "range": [round(b2, 2), 255]},
        ],
    }
