"""
Phase 7.4 — Localized HSV Skin Filtering
========================================

WHAT THIS STAGE DOES

Phase 7.3 hands over five polygon-bounded patches of the SSR-normalized
face. Those polygons are anatomical, not dermal: the forehead polygon still
contains eyebrow hair and stray fringe, the jawline polygon still contains
lip vermilion and facial hair, the cheek polygons can clip an ear or a
strand of hair. Averaging CIELAB over those raw polygons in Phase 8.2 would
mix hair and lip pixels into the skin colour estimate.

This module strips them. Per the manuscript (Stage 3, Feature Selection):

    "Binary masks are applied to isolate the skin pixels within each region
    while excluding all non-region areas. HSV-based skin filtering is then
    performed within each masked region to remove residual non-skin features
    such as eyebrows, eyelashes, hair, eyes, and lips."

Note the word *within*. HSV filtering is applied inside each already-masked
region, one region at a time — not once globally over the whole face. That
is why this is called *localized* HSV filtering in Stage 3 of the System
Architecture, and why the API below takes a single region at a time.

Output: five refined, illumination-normalized regional skin masks, handed to
the Regional Configuration Selector (Phase 7.5).


WHY HSV AND NOT RGB

Skin occupies a compact, well-behaved band of hue regardless of tone: the
melanin/haemoglobin absorption profile puts human skin between roughly 0 deg
and 50 deg on the hue circle for every skin tone from SCC-1 to SCC-6. What
differs between light and deep skin is mostly Value (brightness) and to a
lesser degree Saturation — not Hue. RGB has no such separation; a threshold
that works for SCC-1 fails for SCC-6. Thresholding hue and saturation while
leaving Value nearly unconstrained is what makes one fixed rule tone-fair,
which matters here more than usual because the entire study is about not
disadvantaging darker skin tones.


THRESHOLD PROVENANCE — READ THIS BEFORE YOUR DEFENSE

The manuscript specifies *that* HSV filtering happens but does not state the
numeric thresholds. The defaults below are the widely-cited rule of Kolkur
et al. (2017), "Human Skin Detection Using RGB, HSV and YCbCr Color Models":

    0 deg <= H <= 50 deg,   0.23 <= S <= 0.68

with one documented addition: a Value floor of 0.15, which Kolkur's HSV rule
does not include. It is needed here because the polygon mask zeroes
everything outside the region, and pixels near zero brightness have
numerically unstable hue — without a floor, near-black border pixels get
random hues and some pass the filter. The floor is deliberately low so that
it never discriminates against deeply pigmented skin.

If your panel asks "where did 0.23 come from," the answer is Kolkur et al.
(2017), and you should add that citation to your reference list alongside
the threshold table this module writes to `configs/hsv_skin_thresholds.json`.


THE SSR INTERACTION — THE ONE REAL RISK IN THIS STAGE

SSR takes log(I) - log(F * I) per channel and rescales. Because each of R, G
and B is normalized against its own blurred surround, SSR performs an
implicit per-channel white balance. That shifts hue and inflates saturation
relative to the original photo. A skin rule calibrated on natural images can
therefore over-strip when applied to SSR output.

Two options, both implemented:

  hsv_source="ssr"       Threshold on the SSR patch itself. Literal reading
                         of the manuscript (filtering happens after SSR, on
                         the normalized image). DEFAULT.

  hsv_source="original"  Compute the skin mask on the original pre-SSR crop
                         and apply that mask to the SSR patch. Identical
                         output geometry; the SSR image is still what
                         Phase 8 sees. Only the *pixel selection decision*
                         is made on un-normalized colour.

Run `run_region_qa.py --compare-hsv-source` on a sample before committing to
one. If retention on the SSR path is dramatically lower, say so in your
Discussion chapter and switch — that is a defensible, evidenced methodology
refinement, whereas silently shipping a filter that strips 80% of the
forehead is not.


THE COVERAGE GUARD

A fixed threshold will occasionally fail on a legitimate face — heavy
shadow, unusual white balance, a very saturated background bleeding into the
polygon. Rather than letting those images silently produce a 12-pixel
CIELAB average, filtering runs in tiers:

    tier 1  primary rule        -> status "ok"
    tier 2  relaxed rule        -> status "relaxed"
    tier 3  polygon unfiltered  -> status "geometry_fallback"
    none    nothing usable      -> status "empty"

Every escalation is counted and reported at the Phase 7 checkpoint. A
handful of relaxed regions is normal; if half your dataset is escalating,
the thresholds are wrong for your data and you should recalibrate rather
than accept the fallback.

Design note: this mirrors the graceful-degradation pattern already used in
`data_pipeline.py`, where missing manifest rows are dropped via inner join
instead of raising. Same principle — a robustness measure that is *logged*,
so it stays visible in the write-up instead of quietly changing the results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .regions import (
    MIN_VALID_PIXELS,
    REGION_ORDER,
    STATUS_EMPTY,
    STATUS_GEOMETRY_FALLBACK,
    STATUS_OK,
    STATUS_RELAXED,
    RegionPatch,
)

# --------------------------------------------------------------------------
# Threshold configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HSVSkinThresholds:
    """
    One HSV skin-detection rule, expressed in perceptual units.

    Hue is in degrees (0-360) and saturation/value are fractions (0-1), which
    is how the manuscript and the source literature state them. Conversion to
    OpenCV's uint8 encoding (H 0-179, S/V 0-255) happens in
    `_to_opencv_bounds` so that the numbers stored in the frozen JSON config
    are the ones a reader can check against a paper.
    """

    hue_min_deg: float = 0.0
    hue_max_deg: float = 50.0
    sat_min: float = 0.23
    sat_max: float = 0.68
    val_min: float = 0.15
    val_max: float = 1.0
    #: Optional second hue band for the wrap-around near 360 deg. Some
    #: reddish/ruddy skin lands just below 360 rather than just above 0.
    #: Left disabled by default to keep the primary rule exactly as cited.
    hue_wrap_min_deg: Optional[float] = None

    def _to_opencv_bounds(self):
        """Convert to the (lower, upper) uint8 arrays cv2.inRange expects."""
        h_lo = int(round(self.hue_min_deg / 2.0))          # OpenCV H = deg / 2
        h_hi = int(round(self.hue_max_deg / 2.0))
        s_lo = int(round(self.sat_min * 255))
        s_hi = int(round(self.sat_max * 255))
        v_lo = int(round(self.val_min * 255))
        v_hi = int(round(self.val_max * 255))
        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)

        wrap = None
        if self.hue_wrap_min_deg is not None:
            w_lo = int(round(self.hue_wrap_min_deg / 2.0))
            wrap = (
                np.array([w_lo, s_lo, v_lo], dtype=np.uint8),
                np.array([179, s_hi, v_hi], dtype=np.uint8),
            )
        return lower, upper, wrap


#: Tier 1. Kolkur et al. (2017) HSV rule + documented Value floor.
PRIMARY_THRESHOLDS = HSVSkinThresholds()

#: Tier 2. Widened on all three axes, used only when tier 1 strips a region
#: below the coverage floor. Every use is logged.
RELAXED_THRESHOLDS = HSVSkinThresholds(
    hue_min_deg=0.0,
    hue_max_deg=60.0,
    sat_min=0.12,
    sat_max=0.85,
    val_min=0.08,
    val_max=1.0,
    hue_wrap_min_deg=340.0,
)


@dataclass(frozen=True)
class FilterConfig:
    """Everything about Phase 7.4 that a reviewer might reasonably ask about."""

    primary: HSVSkinThresholds = PRIMARY_THRESHOLDS
    relaxed: HSVSkinThresholds = RELAXED_THRESHOLDS

    #: Minimum fraction of the polygon that must survive, else escalate.
    #: 0.25 is permissive on purpose: the forehead polygon legitimately
    #: contains a lot of hair on some subjects, so a low retention rate is
    #: not automatically a failure.
    min_retention: float = 0.25

    #: Absolute pixel floor. Below this a CIELAB mean is too noisy to trust
    #: regardless of what fraction of the polygon it represents.
    min_pixels: int = MIN_VALID_PIXELS

    #: Morphological cleanup applied inside the polygon.
    open_kernel: int = 3    # removes isolated single-pixel speckle
    close_kernel: int = 5   # fills pores, small specular highlights, stubble gaps

    #: "ssr" (manuscript-literal) or "original" (mask decided on pre-SSR crop).
    hsv_source: str = "ssr"

    def to_json(self, path) -> None:
        """
        Freeze this config to disk.

        Phase 14.1 requires that inference-time constants come from a saved
        artifact rather than being recomputed or re-typed, exactly as the
        K-Means illumination thresholds do. These thresholds are in the same
        category: they were fixed during the study and must not drift between
        training and the demo.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path) -> "FilterConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            primary=HSVSkinThresholds(**raw["primary"]),
            relaxed=HSVSkinThresholds(**raw["relaxed"]),
            min_retention=raw["min_retention"],
            min_pixels=raw["min_pixels"],
            open_kernel=raw["open_kernel"],
            close_kernel=raw["close_kernel"],
            hsv_source=raw["hsv_source"],
        )


DEFAULT_CONFIG = FilterConfig()


# --------------------------------------------------------------------------
# Core masking
# --------------------------------------------------------------------------


def skin_mask_hsv(image_rgb: np.ndarray, thresholds: HSVSkinThresholds) -> np.ndarray:
    """
    Apply one HSV threshold rule to a whole image.

    Args:
        image_rgb:  (H, W, 3) uint8 in RGB order. NOT BGR — the rest of this
                    codebase converts on load (see `global_rgb_features.py`),
                    so the convention is RGB everywhere above the I/O layer.
        thresholds: the rule to apply.

    Returns:
        (H, W) bool array, True where the pixel looks like skin.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got {image_rgb.shape}")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower, upper, wrap = thresholds._to_opencv_bounds()

    mask = cv2.inRange(hsv, lower, upper)
    if wrap is not None:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, wrap[0], wrap[1]))
    return mask.astype(bool)


def _morphological_cleanup(
    mask: np.ndarray,
    geometric_mask: np.ndarray,
    open_kernel: int,
    close_kernel: int,
) -> np.ndarray:
    """
    Tidy a raw threshold mask.

    Opening first (erode then dilate) deletes isolated speckle — single
    pixels that happened to pass the threshold inside a patch of hair.
    Closing second (dilate then erode) fills the small holes that pores,
    stubble and specular highlights punch through otherwise-valid skin.

    The result is re-ANDed with the polygon, because closing dilates and
    could otherwise push the skin mask a few pixels outside the anatomical
    region that Phase 7.3 defined — which would quietly break the claim
    that each region's features come only from that region.
    """
    m = mask.astype(np.uint8)

    if open_kernel and open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    if close_kernel and close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    return m.astype(bool) & geometric_mask


# --------------------------------------------------------------------------
# Per-region filtering (the Phase 7.4 entry point)
# --------------------------------------------------------------------------


def filter_region(
    region: str,
    ssr_image: np.ndarray,
    geometric_mask: np.ndarray,
    config: FilterConfig = DEFAULT_CONFIG,
    reference_image: Optional[np.ndarray] = None,
) -> RegionPatch:
    """
    Run localized HSV skin filtering on one region.

    Args:
        region:          region name, one of REGION_ORDER.
        ssr_image:       (H, W, 3) uint8 RGB, the full SSR-normalized face
                         from Phase 7.2. Not pre-masked — this function
                         applies the mask itself so that HSV is computed on
                         real pixel values rather than on an image that has
                         already had zeros written into it.
        geometric_mask:  (H, W) bool polygon mask for this region, from 7.3.
        config:          thresholds and cleanup settings.
        reference_image: (H, W, 3) uint8 RGB, the ORIGINAL pre-SSR 224x224
                         crop. Required only when config.hsv_source ==
                         "original"; ignored otherwise.

    Returns:
        A fully populated RegionPatch. Never raises on a bad face — a region
        with no usable pixels comes back with status STATUS_EMPTY so the
        caller can decide whether to drop the image or route around it.
    """
    if region not in REGION_ORDER:
        raise ValueError(f"Unknown region {region!r}; expected one of {REGION_ORDER}")

    geometric_mask = geometric_mask.astype(bool)
    n_geom = int(geometric_mask.sum())

    # Masked image handed downstream: SSR pixels inside the region, zeros out.
    masked_image = np.zeros_like(ssr_image)
    masked_image[geometric_mask] = ssr_image[geometric_mask]

    patch = RegionPatch(
        region=region,
        image=masked_image,
        geometric_mask=geometric_mask,
        skin_mask=np.zeros_like(geometric_mask),
        status=STATUS_EMPTY,
        n_geometric_px=n_geom,
        n_valid_px=0,
        meta={"hsv_source": config.hsv_source},
    )

    # Landmark failure or a degenerate polygon: nothing to filter.
    if n_geom < config.min_pixels:
        patch.meta["reason"] = "polygon_too_small"
        return patch

    # Which image decides skin-vs-not-skin. See the SSR interaction note at
    # the top of this file.
    if config.hsv_source == "original":
        if reference_image is None:
            raise ValueError(
                "config.hsv_source == 'original' requires reference_image "
                "(the pre-SSR 224x224 crop from Phase 3)."
            )
        decision_image = reference_image
    elif config.hsv_source == "ssr":
        decision_image = ssr_image
    else:
        raise ValueError(f"hsv_source must be 'ssr' or 'original', got {config.hsv_source!r}")

    # ---- tier 1: primary rule -------------------------------------------
    raw = skin_mask_hsv(decision_image, config.primary) & geometric_mask
    skin = _morphological_cleanup(raw, geometric_mask, config.open_kernel, config.close_kernel)
    n_valid = int(skin.sum())
    retention = n_valid / n_geom

    if n_valid >= config.min_pixels and retention >= config.min_retention:
        patch.skin_mask = skin
        patch.n_valid_px = n_valid
        patch.status = STATUS_OK
        patch.meta["tier"] = "primary"
        return patch

    # ---- tier 2: relaxed rule -------------------------------------------
    raw_r = skin_mask_hsv(decision_image, config.relaxed) & geometric_mask
    skin_r = _morphological_cleanup(raw_r, geometric_mask, config.open_kernel, config.close_kernel)
    n_valid_r = int(skin_r.sum())
    retention_r = n_valid_r / n_geom

    if n_valid_r >= config.min_pixels and retention_r >= config.min_retention:
        patch.skin_mask = skin_r
        patch.n_valid_px = n_valid_r
        patch.status = STATUS_RELAXED
        patch.meta.update(
            {
                "tier": "relaxed",
                "primary_retention": round(retention, 4),
                "relaxed_retention": round(retention_r, 4),
            }
        )
        return patch

    # ---- tier 3: geometry fallback --------------------------------------
    # Both rules stripped the region below usability. Keep the anatomical
    # polygon unfiltered rather than emitting a near-empty region. This is a
    # documented degradation, not a silent one: it is counted at the
    # checkpoint and written to the per-image routing log by Phase 7.5.
    patch.skin_mask = geometric_mask.copy()
    patch.n_valid_px = n_geom
    patch.status = STATUS_GEOMETRY_FALLBACK
    patch.meta.update(
        {
            "tier": "geometry_fallback",
            "primary_retention": round(retention, 4),
            "relaxed_retention": round(retention_r, 4),
        }
    )
    return patch


def filter_all_regions(
    ssr_image: np.ndarray,
    geometric_masks: Dict[str, np.ndarray],
    config: FilterConfig = DEFAULT_CONFIG,
    reference_image: Optional[np.ndarray] = None,
) -> Dict[str, RegionPatch]:
    """
    Run Phase 7.4 across all five regions of one face.

    This is the function Phase 7.3 should call at the end of segmentation,
    and the function whose output goes straight into the Phase 7.5 selector.

    Args:
        ssr_image:       (H, W, 3) uint8 RGB from Phase 7.2.
        geometric_masks: {region_name: (H, W) bool} from Phase 7.3. Missing
                         regions are tolerated and come back STATUS_EMPTY,
                         so a partially-failed mesh does not kill the image.
        config:          see FilterConfig.
        reference_image: original pre-SSR crop; only used when
                         config.hsv_source == "original".

    Returns:
        {region_name: RegionPatch} containing all five REGION_ORDER keys,
        always in canonical order.
    """
    h, w = ssr_image.shape[:2]
    empty = np.zeros((h, w), dtype=bool)

    patches: Dict[str, RegionPatch] = {}
    for name in REGION_ORDER:
        mask = geometric_masks.get(name)
        if mask is None:
            mask = empty
        patches[name] = filter_region(
            region=name,
            ssr_image=ssr_image,
            geometric_mask=mask,
            config=config,
            reference_image=reference_image,
        )
    return patches


# --------------------------------------------------------------------------
# Reporting helper
# --------------------------------------------------------------------------


def summarize(patches: Dict[str, RegionPatch]) -> Dict[str, Dict]:
    """
    Flatten a filtered face into a plain dict, one entry per region.

    Used by the Phase 7 checkpoint script and by the per-image routing log.
    Keeping this separate from the filtering logic means the numbers that
    end up in your Sampling Data write-up come from the same code path that
    produced the masks, not from a second hand-written tally.
    """
    return {
        name: {
            "status": p.status,
            "n_geometric_px": p.n_geometric_px,
            "n_valid_px": p.n_valid_px,
            "retention": round(p.retention, 4),
            "tier": p.meta.get("tier", "none"),
        }
        for name, p in patches.items()
    }
