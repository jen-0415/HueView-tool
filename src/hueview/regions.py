"""
HueView — Shared Regional Contract (used by Phases 7.3, 7.4, 7.5, 8, 9)
=======================================================================

This module holds the vocabulary that every Phase 7+ module agrees on:
region names, their canonical order, the `RegionPatch` container that
7.3 -> 7.4 -> 7.5 -> 8 passes along, and a compact codec for caching
segmentation results to disk.

Nothing here does image processing. It exists so that `hsv_skin_filter.py`
(7.4), `region_selector.py` (7.5) and the Phase 8 feature extractors cannot
silently disagree about what "region 3" means — a disagreement that would
scramble the 15-element CIELAB vector without raising a single error.

CANONICAL ORDER
---------------
The manuscript defines the segmentation order in Stage 3 of Feature
Selection as:

    (1) forehead, (2) left cheek, (3) right cheek, (4) nose bridge, (5) jawline

That order is authoritative here, because it is the order the 15-element
CIELAB Feature Vector (3 values x 5 regions) is built in at Phase 8.2.
Note that Appendix 3's reporting tables (Tables 19-27) list regions in a
slightly different order (Forehead, Left Cheek, Right Cheek, Jawline,
Nose). Reporting order is cosmetic; feature order is not. Use
REGION_ORDER for anything that becomes a feature vector, and
REPORTING_ORDER only when laying out results tables.

LEFT / RIGHT CONVENTION
-----------------------
"Left cheek" means the subject's anatomical left, which appears on the
RIGHT side of the image in a normal front-facing photo. MediaPipe's mesh
is defined in the same subject-anatomical convention, so as long as 7.3
builds its polygons from MediaPipe indices this is automatic. It is
called out explicitly because it is the single easiest thing to flip by
accident, and a flipped pair produces a plausible-looking but wrong
per-region table in Appendix 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Region vocabulary
# --------------------------------------------------------------------------

#: Feature-vector order. Do not reorder — Phase 8.2's 15-element CIELAB
#: vector is built by concatenating regions in exactly this sequence.
REGION_ORDER: Tuple[str, ...] = (
    "forehead",
    "left_cheek",
    "right_cheek",
    "nose_bridge",
    "jawline",
)

#: Order used in the manuscript's results tables (Appendix 3). Cosmetic only.
REPORTING_ORDER: Tuple[str, ...] = (
    "forehead",
    "left_cheek",
    "right_cheek",
    "jawline",
    "nose_bridge",
)

#: Human-readable labels for figures, results tables and the Phase 16 UI.
REGION_DISPLAY: Dict[str, str] = {
    "forehead": "Forehead",
    "left_cheek": "Left Cheek",
    "right_cheek": "Right Cheek",
    "nose_bridge": "Nose Bridge",
    "jawline": "Jawline",
    "full_face": "Full Face (All Five Combined)",
}

#: Integer id per region, 1-5. Zero is reserved for "outside every region".
#: These ids are what the cached label map stores, so changing them
#: invalidates every cached .png on disk.
REGION_IDS: Dict[str, int] = {name: i + 1 for i, name in enumerate(REGION_ORDER)}
ID_TO_REGION: Dict[int, str] = {v: k for k, v in REGION_IDS.items()}

#: Approximate coverage areas from Table 3 of the manuscript, expressed as
#: (y_min, y_max, x_min, x_max) fractions of the face bounding box.
#: Phase 7.3 derives its polygons from MediaPipe landmarks, not from these
#: rectangles — but they are kept here as (a) a sanity check that landmark
#: polygons land roughly where the manuscript says they should, and (b) a
#: degraded fallback for images where MediaPipe returns no mesh.
TABLE3_COVERAGE: Dict[str, Tuple[float, float, float, float]] = {
    "forehead":    (0.00, 0.25, 0.20, 0.80),   # top 25% height, center 60% width
    "left_cheek":  (0.30, 0.65, 0.15, 0.45),   # middle 30-65% height, left 15-45% width
    "right_cheek": (0.30, 0.65, 0.55, 0.85),   # middle 30-65% height, right 55-85% width
    "nose_bridge": (0.35, 0.65, 0.30, 0.70),   # middle 35-65% height, center 30-70% width
    "jawline":     (0.65, 1.00, 0.40, 0.60),   # bottom 20-35% height, center 40-60% width
}

#: Status values a region can carry out of Phase 7.4. Ordered from best to
#: worst; `region_selector` uses this ordering when deciding what to route.
STATUS_OK = "ok"                              # primary HSV rule, healthy coverage
STATUS_RELAXED = "relaxed"                    # relaxed HSV rule was needed
STATUS_GEOMETRY_FALLBACK = "geometry_fallback"  # HSV stripped too much; polygon kept as-is
STATUS_EMPTY = "empty"                        # no usable pixels at all (landmark failure)

STATUS_RANK: Dict[str, int] = {
    STATUS_OK: 0,
    STATUS_RELAXED: 1,
    STATUS_GEOMETRY_FALLBACK: 2,
    STATUS_EMPTY: 3,
}

#: A region below this many valid pixels produces an unstable CIELAB mean.
#: Used as the default floor in both 7.4 and 7.5.
MIN_VALID_PIXELS = 50


# --------------------------------------------------------------------------
# The object that moves between phases
# --------------------------------------------------------------------------

@dataclass
class RegionPatch:
    """
    One anatomical region of one face, at whatever stage of Phase 7 it has
    reached.

    Produced by 7.3 (with `skin_mask` unset), refined by 7.4 (which fills
    `skin_mask`, `status` and the coverage counters), routed by 7.5, and
    consumed by Phase 8.

    Attributes
    ----------
    region:
        One of REGION_ORDER, or "full_face" for the composite built in 7.5.
    image:
        (H, W, 3) uint8 RGB. The SSR-normalized face from Phase 7.2, with
        every pixel outside this region's mask zeroed. Kept at the full
        224x224 frame rather than cropped to a bounding box so that
        EfficientNetB0 receives a consistent input size across regions and
        the spatial position of the patch stays meaningful.
    geometric_mask:
        (H, W) bool. The polygon mask from Phase 7.3, before HSV filtering.
    skin_mask:
        (H, W) bool. Valid skin pixels after Phase 7.4. This is the mask
        Phase 8.2 must average CIELAB over — never `geometric_mask`.
    status:
        One of the STATUS_* constants above.
    n_geometric_px / n_valid_px:
        Pixel counts before and after HSV filtering. Their ratio is the
        retention rate reported in the Phase 7 checkpoint.
    meta:
        Free-form diagnostics (which threshold tier fired, timings, etc.).
        Never read by downstream phases; safe to extend.
    """

    region: str
    image: np.ndarray
    geometric_mask: np.ndarray
    skin_mask: Optional[np.ndarray] = None
    status: str = STATUS_EMPTY
    n_geometric_px: int = 0
    n_valid_px: int = 0
    meta: Dict = field(default_factory=dict)

    @property
    def retention(self) -> float:
        """Fraction of the polygon's pixels that survived HSV filtering."""
        if self.n_geometric_px == 0:
            return 0.0
        return self.n_valid_px / self.n_geometric_px

    @property
    def is_usable(self) -> bool:
        """Whether this region has enough valid pixels to feed Phase 8."""
        return self.status != STATUS_EMPTY and self.n_valid_px >= MIN_VALID_PIXELS

    def masked_pixels(self) -> np.ndarray:
        """
        Return an (N, 3) array of the valid skin pixels only, in RGB.

        This is what Phase 8.2 converts to CIELAB and averages. Returning
        the flat pixel list rather than the masked image avoids the classic
        bug of averaging in the zeroed background, which would drag every
        L* value toward black.
        """
        mask = self.skin_mask if self.skin_mask is not None else self.geometric_mask
        return self.image[mask]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RegionPatch({self.region}, status={self.status}, "
            f"valid={self.n_valid_px}/{self.n_geometric_px} "
            f"({self.retention:.1%}))"
        )


# --------------------------------------------------------------------------
# Cache codec
# --------------------------------------------------------------------------
#
# Running MediaPipe (7.1) is the expensive part of Phase 7 — roughly 15-30 ms
# per image on CPU, which across 29,020 training images and multiple epochs
# is hours of wasted recomputation. SSR (7.2) by contrast is a single
# Gaussian blur and is cheap enough to redo on the fly.
#
# So the recommended caching strategy is: cache the *masks*, recompute SSR.
# The codec below packs all five geometric masks and all five post-HSV skin
# masks into one 224x224 uint8 array that PNG-compresses to a few kilobytes:
#
#     bits 0-2 : region id, 0 = outside all regions, 1-5 = REGION_IDS
#     bit  3   : 1 if this pixel survived Phase 7.4's HSV filter
#
# Regions are assumed non-overlapping. If 7.3 ever produces overlapping
# polygons, the later region in REGION_ORDER wins and a warning is raised.

_SKIN_BIT = 0b1000


def encode_label_map(patches: Dict[str, RegionPatch], shape: Tuple[int, int]) -> np.ndarray:
    """
    Pack a dict of RegionPatches into a single (H, W) uint8 label map.

    Args:
        patches: {region_name: RegionPatch}, as returned by Phase 7.4.
        shape:   (H, W) of the face image, normally (224, 224).

    Returns:
        (H, W) uint8 array suitable for `cv2.imwrite(..., png)`.
    """
    label = np.zeros(shape, dtype=np.uint8)
    for name in REGION_ORDER:
        patch = patches.get(name)
        if patch is None:
            continue
        rid = REGION_IDS[name]
        geom = patch.geometric_mask
        label[geom] = rid
        if patch.skin_mask is not None:
            label[patch.skin_mask] = rid | _SKIN_BIT
    return label


def decode_label_map(
    label: np.ndarray,
    ssr_image: np.ndarray,
) -> Dict[str, RegionPatch]:
    """
    Rebuild RegionPatches from a cached label map plus a freshly recomputed
    SSR image.

    This is the inverse of `encode_label_map`. Statuses are not stored in
    the label map (they live in the per-image routing CSV written by 7.5),
    so decoded patches come back as STATUS_OK when they have pixels and
    STATUS_EMPTY when they do not. If you need exact status replay, join
    against the routing log on filename.
    """
    patches: Dict[str, RegionPatch] = {}
    region_ids = label & 0b0111
    skin_flag = (label & _SKIN_BIT).astype(bool)

    for name in REGION_ORDER:
        rid = REGION_IDS[name]
        geom = region_ids == rid
        skin = geom & skin_flag
        n_geom = int(geom.sum())
        n_valid = int(skin.sum())
        image = np.zeros_like(ssr_image)
        image[geom] = ssr_image[geom]
        patches[name] = RegionPatch(
            region=name,
            image=image,
            geometric_mask=geom,
            skin_mask=skin,
            status=STATUS_OK if n_valid >= MIN_VALID_PIXELS else STATUS_EMPTY,
            n_geometric_px=n_geom,
            n_valid_px=n_valid,
            meta={"source": "cached_label_map"},
        )
    return patches
