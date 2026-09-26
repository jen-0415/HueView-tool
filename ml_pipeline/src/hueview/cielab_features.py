"""
Phase 8.2 -- RGB -> CIELAB Color Branch
Manuscript: Chapter 3, Stage 5 (RGB -> CIELAB Conversion).

    "regional RGB values are first transformed into the intermediate XYZ
    color space via a standard matrix conversion, and subsequently mapped
    into CIELAB coordinates. The L*, a*, and b* values are then extracted and
    averaged exclusively across the valid, unmasked skin pixels of the active
    facial region."

Output per configuration (length = cfg.cielab_dim, set by region_selector 7.5):
    single region : [L*, a*, b*]                           -> 3 values
    full_face     : 5 triples concatenated in REGION_ORDER -> 15 values

How the manuscript uses them:
  * CLASSIFICATION (Fused Classification & Output): "For each facial region,
    the CNN feature vector is combined with the corresponding CIELAB feature
    vector" -> each region's classifier gets that region's 3 values only.
  * The 15 values are the same five triples compiled together ("these
    individual outputs compile into a comprehensive CIELAB Feature Vector",
    Stage 5 -- the "15 regional color features" of the Definition of Terms).
    Stage 6 (undertone) reads them. They are NOT fed to a separate Full Face
    classifier: Full Face combines the five regional predictions (Appendix 3,
    hybrid_classifier.combine_region_predictions).

Conversion: skimage.color.rgb2lab = standard sRGB -> linear RGB -> XYZ ->
CIELAB with the D65 reference white and the 2-degree observer (the sRGB
standard), in floating point (L* 0-100, a*/b* centred on 0). The manuscript
says "standard matrix conversion" without naming a white point -- add
"D65" to Stage 5. NOT
cv2.cvtColor(COLOR_RGB2LAB): OpenCV's 8-bit Lab is rescaled to 0-255 with
a*/b* shifted by +128, which is a display encoding and would corrupt Phase 9's
hue angle atan2(b*, a*).

Averaging is done over the skin mask only (RegionPatch.masked_pixels), never
over the zeroed background, which would drag every L* toward black.

CHANGES FROM THE PREVIOUS VERSION
  * A configuration with NO usable pixels now returns None instead of a
    (0, 0, 0) vector. Previously a single-region config (or a full_face with
    all five regions empty) fell back to zeros -- pure black in CIELAB, which
    biases toward SCC-6. Returning None matches cnn_features.prepare_patch,
    so 8.1 and 8.2 skip exactly the same examples.
  * impute_policy now defaults to the policy the selector recorded on the
    ConfigurationOutput (cfg.meta["impute_policy"]), so the policy chosen in
    7.5 is the one actually applied.
  * Added a --smoke self-test with reference sRGB -> CIELAB values.
  * D65 / 2-degree observer passed explicitly (same values as before -- they
    are skimage's defaults -- but now visible and citable).

Function names are unchanged (region_cielab_mean, build_cielab_vector).

Usage (run from ml_pipeline/):
    python -m src.hueview.cielab_features --smoke

Requires: pip install scikit-image
"""

from __future__ import annotations

import argparse
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

LAB_DIM = 3


def region_cielab_mean(
    patch: RegionPatch,
    source_image: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Mean [L*, a*, b*] over one region's valid skin pixels.

    Args:
        patch:        the region's RegionPatch, providing the skin mask.
        source_image: optional (H, W, 3) uint8 RGB override. If given, pixel
                      VALUES are sampled from this image at the patch's skin
                      mask instead of patch.image (the SSR pixels). Diagnostic
                      only: compares manuscript-literal SSR-sourced CIELAB
                      against pre-SSR colour at the same mask, because SSR's
                      per-channel rescaling shifts hue (see hsv_skin_filter.py).
                      Leave as None for the manuscript pipeline.

    Returns:
        float64 array [L*, a*, b*], or None if the region has no usable pixels.
    """
    if not patch.is_usable:
        return None

    if source_image is not None:
        mask = patch.skin_mask if patch.skin_mask is not None else patch.geometric_mask
        pixels = source_image[mask]
    else:
        pixels = patch.masked_pixels()  # (N, 3) uint8 RGB from the SSR image

    if pixels.size == 0:
        return None

    rgb_float = pixels.astype(np.float64) / 255.0
    # rgb2lab expects an image; (N, 1, 3) treats each pixel as a 1x1 image.
    lab = rgb2lab(rgb_float.reshape(-1, 1, 3),
                  illuminant="D65", observer="2").reshape(-1, 3)
    return lab.mean(axis=0)


def build_cielab_vector(
    cfg: ConfigurationOutput,
    impute_policy: Optional[str] = None,
    original_rgb: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Assemble the CIELAB feature vector for one ConfigurationOutput.

    Length is cfg.cielab_dim: 3 for a single region, 15 for full_face
    (five triples concatenated in cfg.component_order(), i.e. REGION_ORDER).

    Args:
        impute_policy: how to fill a full_face component region that has no
                       usable pixels. None (default) = use the policy the
                       selector recorded in cfg.meta, else IMPUTE_FACE_MEAN.
                         IMPUTE_FACE_MEAN -- mean of this face's other usable
                                             regions (regions of one face are
                                             strongly correlated in colour).
                         IMPUTE_NAN       -- NaN; the training loop decides.
                         IMPUTE_ZEROS     -- 0.0; NOT recommended (pure black).
        original_rgb:  optional pre-SSR image, see region_cielab_mean.

    Returns:
        float64 vector of length cfg.cielab_dim, or None when NO component
        region is usable (nothing to impute from). The caller skips that
        example, exactly like cnn_features.prepare_patch returning None.
    """
    if impute_policy is None:
        impute_policy = cfg.meta.get("impute_policy", IMPUTE_FACE_MEAN)

    means: Dict[str, Optional[np.ndarray]] = {
        name: region_cielab_mean(patch, source_image=original_rgb)
        for name, patch in cfg.components.items()
    }
    available = [v for v in means.values() if v is not None]

    # Nothing usable at all: never invent a colour. This covers a single-region
    # config whose region is empty, and a full_face whose five regions all are.
    if not available:
        return None

    if impute_policy == IMPUTE_FACE_MEAN:
        fallback = np.mean(available, axis=0)
    elif impute_policy == IMPUTE_NAN:
        fallback = np.full(LAB_DIM, np.nan)
    elif impute_policy == IMPUTE_ZEROS:
        fallback = np.zeros(LAB_DIM)
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


# ---------------------------------------------------------------------------
# Smoke test (no dataset needed)
# ---------------------------------------------------------------------------
def _smoke() -> int:
    from .regions import REGION_ORDER, STATUS_EMPTY, STATUS_OK
    from .region_selector import (
        FULL_FACE,
        RegionalConfigurationSelector,
        assert_feature_shapes,
    )

    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [ok]   {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name} {detail}")

    H = W = 224

    def make_patch(region, color, box, usable=True):
        """A RegionPatch whose skin pixels are all `color`, zeros elsewhere."""
        y0, y1, x0, x1 = box
        geom = np.zeros((H, W), bool)
        geom[y0:y1, x0:x1] = True
        skin = geom.copy() if usable else np.zeros((H, W), bool)
        img = np.zeros((H, W, 3), np.uint8)
        img[skin] = color
        n = int(skin.sum())
        return RegionPatch(region=region, image=img, geometric_mask=geom,
                           skin_mask=skin,
                           status=STATUS_OK if usable else STATUS_EMPTY,
                           n_geometric_px=int(geom.sum()), n_valid_px=n)

    boxes = {
        "forehead": (10, 50, 50, 170), "left_cheek": (80, 140, 130, 190),
        "right_cheek": (80, 140, 30, 90), "nose_bridge": (80, 140, 95, 125),
        "jawline": (160, 210, 90, 135),
    }
    colors = {  # five distinct skin-like colours
        "forehead": (224, 172, 105), "left_cheek": (198, 134, 66),
        "right_cheek": (141, 85, 36), "nose_bridge": (255, 219, 172),
        "jawline": (92, 51, 23),
    }

    print("\n== standard sRGB -> CIELAB reference values (D65) ==")
    white = region_cielab_mean(make_patch("forehead", (255, 255, 255), boxes["forehead"]))
    check("white -> L*=100, a*=0, b*=0",
          np.allclose(white, [100, 0, 0], atol=0.01), str(white))
    red = region_cielab_mean(make_patch("forehead", (255, 0, 0), boxes["forehead"]))
    check("sRGB red -> (53.24, 80.09, 67.20)",
          np.allclose(red, [53.24, 80.09, 67.20], atol=0.02), str(red))
    check("a*/b* centred on 0 (not OpenCV's +128 encoding)",
          abs(white[1]) < 0.01 and abs(white[2]) < 0.01)

    print("\n== mean over skin pixels only ==")
    p = make_patch("forehead", colors["forehead"], boxes["forehead"])
    one = region_cielab_mean(p)
    ref = rgb2lab(np.array(colors["forehead"], float).reshape(1, 1, 3) / 255).ravel()
    check("zeroed background does not pull L* toward black",
          np.allclose(one, ref, atol=1e-6), f"{one} vs {ref}")

    print("\n== configurations from the real 7.5 selector ==")
    patches = {r: make_patch(r, colors[r], boxes[r]) for r in REGION_ORDER}
    selector = RegionalConfigurationSelector(skip_unusable=False)
    outs = selector.route_all(patches, image_id="synthetic")
    vecs = {c: build_cielab_vector(o) for c, o in outs.items()}

    check("6 configurations routed", len(outs) == 6, str(list(outs)))
    check("single regions -> 3 values",
          all(vecs[r].shape == (3,) for r in REGION_ORDER))
    check("full_face -> 15 values", vecs[FULL_FACE].shape == (15,))
    check("full_face = the 5 regional triples in REGION_ORDER",
          np.allclose(vecs[FULL_FACE],
                      np.concatenate([vecs[r] for r in REGION_ORDER])))
    fused = {r: assert_feature_shapes(outs[r], np.zeros(1280), vecs[r])
             for r in REGION_ORDER}
    check("Phase 8 checkpoint: every region fuses to 1280 + 3 = 1283",
          all(v == 1283 for v in fused.values()), str(fused))

    print("\n== missing regions ==")
    patches_miss = dict(patches)
    patches_miss["jawline"] = make_patch("jawline", colors["jawline"],
                                         boxes["jawline"], usable=False)
    outs_m = selector.route_all(patches_miss, image_id="missing_jaw")
    ff = build_cielab_vector(outs_m[FULL_FACE])
    others = np.mean([vecs[r] for r in REGION_ORDER if r != "jawline"], axis=0)
    check("full_face: missing jawline imputed from the other 4 (face_mean)",
          np.allclose(ff[12:15], others), f"{ff[12:15]} vs {others}")
    check("full_face records the imputation",
          outs_m[FULL_FACE].imputed_regions == ["jawline"],
          str(outs_m[FULL_FACE].imputed_regions))
    ff_nan = build_cielab_vector(outs_m[FULL_FACE], impute_policy=IMPUTE_NAN)
    check("impute_policy='nan' leaves NaN in the jawline slot",
          np.isnan(ff_nan[12:15]).all() and not np.isnan(ff_nan[:12]).any())

    check("BUG FIX: empty single region -> None (was (0,0,0) = black)",
          build_cielab_vector(outs_m["jawline"]) is None,
          str(build_cielab_vector(outs_m["jawline"])))

    all_empty = {r: make_patch(r, colors[r], boxes[r], usable=False)
                 for r in REGION_ORDER}
    outs_e = selector.route_all(all_empty, image_id="all_empty")
    check("BUG FIX: full_face with all regions empty -> None",
          build_cielab_vector(outs_e[FULL_FACE]) is None)

    print("\n== pre-SSR diagnostic switch ==")
    orig = np.full((H, W, 3), 128, np.uint8)
    d = region_cielab_mean(patches["forehead"], source_image=orig)
    grey = rgb2lab(np.full((1, 1, 3), 128 / 255)).ravel()
    check("source_image samples the other image at the same mask",
          np.allclose(d, grey, atol=1e-6))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 8.2 CIELAB color branch")
    ap.add_argument("--smoke", action="store_true",
                    help="run self-checks without any dataset")
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(_smoke())
    ap.print_help()
