"""
Phase 9 -- Undertone Descriptor (Exploratory, no training needed)

Per the manuscript (Stage 6): a deterministic, rule-based procedure reusing
the CIELAB values already computed in Phase 8.2 -- no new input, no
training. There is no ground truth for undertone in the STW dataset, so
this is reported descriptively (e.g. for cosmetic-shade matching) and
explicitly excluded from performance-metric evaluation.

    h_ab = atan2(b*, a*), mapped to 0-360 deg (Schloss et al., 2018)

    Warm:    h_ab > 65 deg   (yellow-dominant)
    Neutral: 55 <= h_ab <= 65 deg
    Cool:    h_ab < 55 deg   (red/pink-dominant)

Thresholds are per the manuscript's cited Pantone SkinTone Guide
(Sirisayan, 2022).

For a 5-region vector (Full Face), the five per-region labels are combined
by majority vote into a single Majority Undertone Label. The manuscript
describes the majority vote itself but does not specify a tie-break rule;
the build plan does -- "whichever regional descriptor is numerically
closest to the 60 deg reference angle" -- implemented here as: among the
five individual regions, find the one whose hue angle is nearest to 60 deg
and use ITS label as the tie-breaking choice. This is a build-plan
addition, not manuscript-specified -- worth confirming with your adviser
alongside the other flagged discrepancies, since a genuine tie (e.g. 2
Warm / 2 Cool / 1 Neutral) is a real possibility with 5 votes across 3
categories, not just a theoretical edge case.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import numpy as np

from .region_selector import ConfigurationOutput

WARM = "Warm"
NEUTRAL = "Neutral"
COOL = "Cool"

REFERENCE_ANGLE_DEG = 60.0
WARM_THRESHOLD_DEG = 65.0
COOL_THRESHOLD_DEG = 55.0


def hue_angle_deg(a: float, b: float) -> float:
    """h_ab = atan2(b*, a*), mapped to 0-360 deg (Schloss et al., 2018)."""
    return float(np.degrees(np.arctan2(b, a)) % 360)


def classify_undertone(hue_deg: float) -> str:
    """
    Warm if >65 deg, Cool if <55 deg, Neutral in between -- per the
    manuscript's Pantone SkinTone Guide-derived thresholds (Sirisayan, 2022).

    Hue is circular: a value in [340, 360) is numerically close to 360 but
    perceptually adjacent to 0 (the red/"Cool" axis), not to 65+ ("Warm").
    Without this correction, measurement noise that pushes a near-zero true
    hue slightly negative (e.g. -0.1 deg) gets wrapped to 359.9 deg by
    hue_angle_deg()'s % 360 normalization and then misclassified as Warm
    purely as a wraparound artifact -- not because the underlying colour is
    actually yellow-dominant. The 340 deg cutoff mirrors the wrap
    convention hsv_skin_filter.py's RELAXED_THRESHOLDS already uses
    (hue_wrap_min_deg=340) for the same reason, applied there to skin-pixel
    detection rather than undertone classification.
    """
    if hue_deg >= 340.0:
        hue_deg -= 360.0
    if hue_deg > WARM_THRESHOLD_DEG:
        return WARM
    if hue_deg < COOL_THRESHOLD_DEG:
        return COOL
    return NEUTRAL


def majority_undertone(per_region_hues: Dict[str, float]) -> Tuple[str, bool]:
    """
    Majority-vote across regions' individual undertone labels.

    Returns (majority_label, was_tied). On a tie, per the build plan:
    "whichever regional descriptor is numerically closest to the 60 deg
    reference angle" -- among ALL regions (not just the tied categories),
    the one whose hue is nearest 60 deg breaks the tie with its own label.
    """
    labels = {name: classify_undertone(hue) for name, hue in per_region_hues.items()}
    counts = Counter(labels.values())
    top_count = max(counts.values())
    leaders = [label for label, c in counts.items() if c == top_count]

    if len(leaders) == 1:
        return leaders[0], False

    closest_region = min(per_region_hues, key=lambda n: abs(per_region_hues[n] - REFERENCE_ANGLE_DEG))
    return labels[closest_region], True


def compute_undertone_descriptor(cfg: ConfigurationOutput, cielab_vector: np.ndarray) -> dict:
    """
    Build the undertone descriptor for one ConfigurationOutput.

    For a single-region config (cielab_dim == 3): one hue angle, one label,
    no vote needed. For full_face (cielab_dim == 15): one hue angle + label
    per component region, combined via majority vote (with the build-plan
    tie-break) into a single Majority Undertone Label -- matching Phase
    14's inference output shape ({"regions": {...}, "majority_undertone": ...}).

    Never scored against ground truth -- STW has no undertone labels.
    """
    order = cfg.component_order()
    n_regions = len(order)

    if cielab_vector.shape[0] != 3 * n_regions:
        raise ValueError(
            f"[{cfg.config}] CIELAB vector is {cielab_vector.shape[0]}-d, "
            f"expected {3 * n_regions} for {n_regions} region(s)."
        )

    per_region = cielab_vector.reshape(n_regions, 3)
    hues: Dict[str, float] = {}
    labels: Dict[str, str] = {}

    for name, (L, a, b) in zip(order, per_region):
        hue = hue_angle_deg(a, b)
        hues[name] = hue
        labels[name] = classify_undertone(hue)

    if n_regions == 1:
        only = order[0]
        return {
            "regions": {only: {"hue_deg": hues[only], "undertone": labels[only]}},
            "majority_undertone": labels[only],
            "was_tied": False,
        }

    majority_label, was_tied = majority_undertone(hues)
    return {
        "regions": {name: {"hue_deg": hues[name], "undertone": labels[name]} for name in order},
        "majority_undertone": majority_label,
        "was_tied": was_tied,
    }
