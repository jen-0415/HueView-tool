"""
Phase 8.1 -- CNN Feature Branch (HueView hybrid model)
Manuscript: Chapter 3, "Hybrid Skin Tone Feature Extraction & Classification",
            Stage 4 (EfficientNet CNN Feature Extraction).

Turns ONE region's SSR-normalized, HSV-filtered skin patch into a CNN feature
vector with EfficientNetB0. Regions are processed one at a time ("fed
sequentially -- one region at a time -- into the EfficientNetB0 backbone").
Phase 8.2 (CIELAB, 3 values for the same region) and 8.3 (concatenate ->
fully connected -> 6-class softmax, per region) consume this output.

-------------------------------------------------------------------------------
DESIGN DECISION: one SHARED EfficientNetB0 backbone, one classifier per region
-------------------------------------------------------------------------------
All five regions go through the same EfficientNetB0 (same architecture, same
ImageNet weights). Each region then gets its OWN fused classifier (8.3), so
predictions are still produced "for each region independently".

Grounding in the manuscript:
  * Appendix 1, Model Setup: "Both models will use the identical EfficientNetB0
    backbone" -- one backbone, shared with the baseline configuration.
  * Stage 4: patches are fed "into the EfficientNetB0 backbone" (singular).
  * Definition of Terms: EfficientNetB0 is "pretrained on the ImageNet dataset".

Why not one backbone per region:
  1. Parameters: shared = ~4.05M backbone params; five separate backbones =
     ~20.2M. Training runs on CPU only.
  2. Data: every region comes from the same images, so a per-region backbone
     sees no extra data -- only a smaller window of the same N images -- and
     overfits more easily.
  3. Fairness of comparison: the baseline has one EfficientNetB0. A shared
     backbone keeps HueView's CNN capacity equal, so any improvement is due to
     SSR / regional segmentation / CIELAB, which is what the study tests.
  4. With a frozen backbone (default here) "separate" backbones would be five
     identical copies anyway; the choice only matters if the backbone is
     fine-tuned. MATCH THE BASELINE: if the Phase 6 baseline fine-tunes
     EfficientNetB0, set trainable=True here too.

-------------------------------------------------------------------------------
Non-skin pixels -- follows the manuscript
-------------------------------------------------------------------------------
Stage 3: binary masks are applied "zeroing out all background pixels outside
the targeted zones". So pixels outside the region AND pixels rejected by the
HSV filter are set to 0 (fill="zero", the default). fill="mean" is kept only
as an optional experiment; using it would require a manuscript change.

Implementation detail not specified by the manuscript: the patch is cropped
to the mask's bounding box and letterboxed back to 224x224, so small regions
(nose bridge, jawline) fill the CNN's input instead of occupying a few pixels
of a mostly black 224x224 frame.

-------------------------------------------------------------------------------
Full Face -- NOT a CNN configuration
-------------------------------------------------------------------------------
Appendix 3: "the Full Face setup formed by combining the predictions of all
five regions". Full Face is therefore built AFTER 8.3 from the five regional
predictions; it does not get its own CNN input or a 6400-dim feature vector.

-------------------------------------------------------------------------------
Checkpoint for 8.3
-------------------------------------------------------------------------------
  every region : CNN 1280 + CIELAB 3 = 1283  -> Dense -> softmax(6)

Usage (run from ml_pipeline/, same as the other src.hueview modules):
    python -m src.hueview.cnn_features --smoke            # no dataset, no download
    python -m src.hueview.cnn_features --smoke --imagenet # also checks real weights
"""
from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The five regions of Chapter 3, Table 3. Keep these keys identical to
# region_selector.py (Phase 7.5) so every phase uses the same names.
REGIONS = ("forehead", "left_cheek", "right_cheek", "nose_bridge", "jawline")

INPUT_SIZE = 224          # manuscript: faces resized to 224 x 224
CNN_DIM = 1280            # EfficientNetB0 global-average-pooled output
LAB_DIM = 3               # Phase 8.2: [L*, a*, b*] per region
FUSED_DIM = CNN_DIM + LAB_DIM
MIN_VALID_PIXELS = 50     # patches with fewer skin pixels are skipped


# ---------------------------------------------------------------------------
# Patch preparation
# ---------------------------------------------------------------------------
def prepare_patch(
    rgb: np.ndarray,
    mask: np.ndarray,
    size: int = INPUT_SIZE,
    fill: str = "zero",
    crop: bool = True,
    min_valid: int = MIN_VALID_PIXELS,
) -> Optional[np.ndarray]:
    """
    Turn one region's SSR image + skin mask into an EfficientNetB0 input.

    Args:
        rgb:  HxWx3 uint8 in RGB order -- the SSR-normalized face.
              NOTE: cv2.imread returns BGR; convert with cv2.cvtColor first.
        mask: HxW, nonzero = valid skin pixel (region mask AND HSV skin mask).
        size: output side length.
        fill: "zero" (manuscript) or "mean" (experimental) for non-skin pixels.
        crop: crop to the mask's bounding box before resizing (see docstring).
        min_valid: return None if fewer skin pixels than this.

    Returns:
        float32 (size, size, 3), values 0..255. Keras' EfficientNetB0 rescales
        internally, so do NOT divide by 255.
        None if the patch has too few skin pixels (caller should log + skip).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match image {rgb.shape[:2]}")

    valid = mask.astype(bool)
    if int(valid.sum()) < min_valid:
        return None

    patch = rgb.astype(np.float32)
    pmask = valid
    if crop:
        ys, xs = np.nonzero(valid)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch = patch[y0:y1, x0:x1]
        pmask = valid[y0:y1, x0:x1]

    if fill == "zero":
        fill_color = np.zeros(3, np.float32)
    elif fill == "mean":
        fill_color = patch[pmask].mean(axis=0)
    else:
        raise ValueError(f"fill must be 'zero' or 'mean', got {fill!r}")
    patch = patch.copy()
    patch[~pmask] = fill_color

    # Letterbox to size x size, keeping aspect ratio (cheek and forehead
    # shapes differ a lot; stretching would distort skin texture).
    h, w = patch.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(patch, (nw, nh), interpolation=interp)

    out = np.empty((size, size, 3), np.float32)
    out[:] = fill_color
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_backbone(weights: Optional[str] = "imagenet",
                   trainable: bool = False,
                   input_size: int = INPUT_SIZE):
    """
    The shared EfficientNetB0: (size, size, 3) -> (1280,).
    Build it ONCE and reuse it for every region.
    """
    from tensorflow import keras

    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
        pooling="avg",
    )
    base.trainable = trainable
    return base


def build_cnn_branch(region: str, backbone, input_size: int = INPUT_SIZE):
    """
    Wrap the shared backbone for one region: "<region>_patch" -> (batch, 1280).

    Pass the SAME backbone object for every region -- that is what makes the
    weights shared.
    """
    from tensorflow import keras

    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}. Options: {list(REGIONS)}")
    inp = keras.Input((input_size, input_size, 3), name=f"{region}_patch")
    # training=False keeps BatchNorm in inference mode, which is standard
    # Keras transfer-learning practice even if the backbone is unfrozen later.
    feat = backbone(inp, training=False)
    return keras.Model(inp, feat, name=f"cnn_branch_{region}")


def extract_features(backbone, patches: np.ndarray,
                     batch_size: int = 32) -> np.ndarray:
    """
    Run a frozen backbone over prepared patches -> (N, 1280) float32.

    With a frozen backbone the CNN vectors never change during training, so
    computing them once and saving to .npy makes 8.3 training on CPU fast.
    Caveat: training-set augmentation (flip / rotation / color jitter, per
    Appendix 1) must then be applied BEFORE extraction, as fixed augmented
    copies, instead of randomly each epoch.
    """
    return backbone.predict(patches, batch_size=batch_size, verbose=0).astype(
        np.float32)


# ---------------------------------------------------------------------------
# Smoke test / checkpoint (no dataset needed)
# ---------------------------------------------------------------------------
def _smoke(use_imagenet: bool) -> int:
    import tensorflow as tf

    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [ok]   {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name} {detail}")

    print("\n== prepare_patch ==")
    rng = np.random.default_rng(0)
    img = rng.integers(1, 256, (224, 224, 3), dtype=np.uint8)  # no true zeros
    mask = np.zeros((224, 224), np.uint8)
    mask[40:100, 60:180] = 1            # a forehead-like region
    mask[60:70, 100:120] = 0            # a hole (HSV-rejected pixels)

    p = prepare_patch(img, mask)
    check("output shape (224,224,3)", p is not None and p.shape == (224, 224, 3),
          f"got {None if p is None else p.shape}")
    check("dtype float32", p is not None and p.dtype == np.float32)
    check("range stays 0..255 (no /255)",
          p is not None and 0 <= p.min() and p.max() <= 255)
    check("default fill = zero (manuscript): padding is black",
          p is not None and np.allclose(p[0, 0], 0))

    nc = prepare_patch(img, mask, crop=False)
    outside = ~mask.astype(bool)
    check("crop=False keeps 224 frame, zeroes everything outside the mask",
          nc is not None and np.allclose(nc[outside], 0)
          and np.allclose(nc[mask.astype(bool)], img[mask.astype(bool)]))

    pm = prepare_patch(img, mask, fill="mean")
    skin_mean = img[mask.astype(bool)].astype(np.float32).mean(axis=0)
    check("fill='mean' option still works",
          pm is not None and np.allclose(pm[0, 0], skin_mean, atol=1e-3))

    check("empty mask -> None", prepare_patch(img, np.zeros_like(mask)) is None)
    tiny = np.zeros_like(mask); tiny[0:3, 0:3] = 1
    check("mask below MIN_VALID_PIXELS -> None", prepare_patch(img, tiny) is None)
    try:
        prepare_patch(img, mask[:10])
        check("mismatched mask raises", False)
    except ValueError:
        check("mismatched mask raises", True)

    print("\n== model shapes (checkpoint for 8.3) ==")
    backbone = build_backbone("imagenet" if use_imagenet else None)
    branches = {r: build_cnn_branch(r, backbone) for r in REGIONS}

    check("5 region branches built", len(branches) == 5)
    check("every region -> (None, 1280)",
          all(tuple(m.output.shape) == (None, CNN_DIM) for m in branches.values()),
          str({r: tuple(m.output.shape) for r, m in branches.items()}))
    check("all branches use the SAME backbone object",
          all(m.layers[1] is backbone for m in branches.values()))
    check("backbone frozen (0 trainable weights)",
          len(backbone.trainable_weights) == 0)

    batch = np.stack([p, p])
    outs = {r: m.predict(batch, verbose=0) for r, m in branches.items()}
    check("forward pass -> (2,1280) per region",
          all(o.shape == (2, CNN_DIM) for o in outs.values()))
    check("same patch -> same vector in every region (weights shared)",
          all(np.allclose(outs["forehead"], o, atol=1e-5) for o in outs.values()))
    check("no NaN/Inf", all(np.isfinite(o).all() for o in outs.values()))

    feats = extract_features(backbone, batch)
    check("extract_features -> (2,1280) float32",
          feats.shape == (2, CNN_DIM) and feats.dtype == np.float32)
    check("fused dim for 8.3 = 1283", FUSED_DIM == 1283)

    n = backbone.count_params()
    print("\n== summary ==")
    print(f"  per region: CNN {CNN_DIM} + CIELAB {LAB_DIM} = {FUSED_DIM} -> Dense -> softmax(6)")
    print(f"  backbone params (shared, counted once): {n:,}")
    print(f"  if one backbone per region instead:     {n * len(REGIONS):,}")
    print(f"\n{passed} passed, {failed} failed  (TF {tf.__version__}, "
          f"weights={'imagenet' if use_imagenet else 'random'})")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 8.1 CNN feature branch")
    ap.add_argument("--smoke", action="store_true",
                    help="run shape/logic checks without any dataset")
    ap.add_argument("--imagenet", action="store_true",
                    help="use real ImageNet weights (downloads ~16 MB once)")
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(_smoke(args.imagenet))
    ap.print_help()
