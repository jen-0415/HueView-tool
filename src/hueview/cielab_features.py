"""
Phase 8.2 -- RGB -> CIELAB Color Branch

Per the manuscript (Stage 5): "regional RGB values are first transformed
into the intermediate XYZ color space via a standard matrix conversion, and
subsequently mapped into CIELAB coordinates. The L*, a*, and b* values are
then extracted and averaged exclusively across the valid, unmasked skin
pixels of the active facial region."

Uses skimage.color.rgb2lab rather than cv2.cvtColor(..., COLOR_RGB2LAB):
cv2's Lab conversion is quantized to uint8 (L rescaled to 0-255, a/b shifted
by +128) for 8-bit image display, not for numerical feature work --
averaging over that quantized, shifted encoding and then trying to recover
standard L*/a*/b* units for Phase 9's hue-angle formula (atan2(b*, a*),
which assumes a*/b* centered on 0) introduces avoidable precision loss.
skimage.color.rgb2lab returns true floating-point CIELAB (L* in 0-100,
a*/b* roughly -128 to 127) directly from float RGB in [0, 1], which is what
both this stage and Phase 9 actually need.

Handles the Full Face imputation policy region_selector.py (7.5) flagged
but explicitly did not implement ("the selector routes, it does not
compute features") -- this module is that computation.

Requires: pip install scikit-image
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from skimage.color import rgb2lab

from .regions import RegionPatch
from .region_selector import (
    ConfigurationOutput,
    IMPUTE_FACE_MEAN,
    IMPUTE_NAN,
    IMPUTE_ZEROS,
)


def region_cielab_mean(
    patch: RegionPatch,
    source_image: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Mean [L*, a*, b*] over one region's valid skin pixels.

    Args:
        patch:        the region's RegionPatch, providing the skin mask.
        source_image: optional (H, W, 3) uint8 RGB override. If given,
                      pixel VALUES are sampled from this image at the
                      patch's skin_mask locations, instead of
                      patch.image (which holds the SSR-normalized
                      pixels). This exists to compare the
                      manuscript-literal SSR-sourced CIELAB features
                      against features computed from the original,
                      pre-SSR colour at the same mask -- mirroring the
                      hsv_source comparison already used to validate
                      Phase 7.4's mask decision, because SSR's
                      per-channel rescaling is known to distort hue
                      severely (see hsv_skin_filter.py's own docstring),
                      and that distortion lands directly in these color
                      features, not just in a filtering threshold.

    Returns None if the region has no usable pixels -- the caller
    (build_cielab_vector) decides what to do about a hole in the vector,
    per its impute_policy argument.
    """
    if not patch.is_usable:
        return None

    if source_image is not None:
        mask = patch.skin_mask if patch.skin_mask is not None else patch.geometric_mask
        pixels = source_image[mask]
    else:
        pixels = patch.masked_pixels()  # (N, 3) uint8 RGB, from the SSR image

    if pixels.size == 0:
        return None

    rgb_float = pixels.astype(np.float64) / 255.0
    # rgb2lab expects an (H, W, 3) image; reshape the flat pixel list to
    # (N, 1, 3) so each pixel is treated as its own 1x1 "image".
    lab = rgb2lab(rgb_float.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab.mean(axis=0)  # [L*, a*, b*]


def build_cielab_vector(
    cfg: ConfigurationOutput,
    impute_policy: str = IMPUTE_FACE_MEAN,
    original_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Assemble the CIELAB feature vector for one ConfigurationOutput.

    Length is cfg.cielab_dim: 3 for a single-region config, 15 for
    full_face (5 regions x 3, concatenated in cfg.component_order() --
    REGION_ORDER -- per the manuscript's "3 values x 5 regions").

    Args:
        original_rgb: optional (H, W, 3) uint8 RGB, the pre-SSR 224x224
                      crop for this same image. When given, every
                      component region's CIELAB mean is computed from
                      this image's pixels at the region's skin mask,
                      instead of the SSR-normalized pixels. See
                      region_cielab_mean for why this matters.

    A component region with no usable pixels is filled according to
    impute_policy:
        IMPUTE_FACE_MEAN -- mean of this same face's OTHER surviving
                            regions (the mildest assumption -- regions of
                            one face are strongly correlated in color).
                            DEFAULT.
        IMPUTE_NAN       -- np.nan, let the training loop decide (e.g.
                            drop the example, or a masked loss).
        IMPUTE_ZEROS     -- 0.0 -- NOT recommended; (0,0,0) in CIELAB is
                            pure black and would bias the classifier
                            toward the darkest SCC class.
    """
    means: Dict[str, Optional[np.ndarray]] = {
        name: region_cielab_mean(patch, source_image=original_rgb)
        for name, patch in cfg.components.items()
    }

    available = [v for v in means.values() if v is not None]

    if impute_policy == IMPUTE_FACE_MEAN:
        fallback = np.mean(available, axis=0) if available else np.zeros(3)
    elif impute_policy == IMPUTE_NAN:
        fallback = np.full(3, np.nan)
    elif impute_policy == IMPUTE_ZEROS:
        fallback = np.zeros(3)
    else:
        raise ValueError(f"Unknown impute_policy {impute_policy!r}")

    ordered = [
        means[name] if means[name] is not None else fallback
        for name in cfg.component_order()
    ]
    vector = np.concatenate(ordered)

    if vector.shape[0] != cfg.cielab_dim:
        raise ValueError(
            f"[{cfg.config}] built a {vector.shape[0]}-d CIELAB vector, "
            f"expected {cfg.cielab_dim}."
        )
    return vector
