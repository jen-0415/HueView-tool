"""
Phase 8.1 -- CNN Feature Branch (HueView hybrid model)
Manuscript: Chapter 3, Stage 4 (EfficientNet CNN Feature Extraction);
            Definition of Terms, "EfficientNetB0"; Appendix 1, Model Setup.

Turns ONE region's SSR-normalized, HSV-filtered skin patch into a 1280-d
region-specific CNN feature vector. Phase 8.2 computes that region's
[L*, a*, b*]; Phase 8.3 concatenates the two and classifies.

-------------------------------------------------------------------------------
What the manuscript specifies, and how this file follows it
-------------------------------------------------------------------------------
* "EfficientNetB0 ... pretrained on the ImageNet dataset" (Definition of Terms)
      -> keras EfficientNetB0, weights="imagenet", include_top=False,
         global average pooling -> 1280-d vector.
* "isolated, illumination-adjusted skin patches ... fed sequentially -- one
  region at a time -- into the EfficientNetB0 backbone" (Stage 4)
      -> one region per call; input is the region's SSR image with every
         pixel outside its skin mask zeroed.
* "binary pixel masks ... zeroing out all background pixels outside the
  targeted zones" (Stage 3)
      -> fill="zero". The patch stays in its full 224 x 224 frame (crop=False),
         which is also the RegionPatch contract in regions.py. The manuscript
         never mentions cropping, so cropping is an opt-in experiment only.
* "learns fine-grained skin color distributions and localized texture
  patterns from each segmented region independently" (Definition of Terms;
  Stage 4)
      -> EACH REGION GETS ITS OWN EfficientNetB0, initialised from the same
         ImageNet weights and trained (fine-tuned) on that region's patches
         only. build_backbone() therefore returns a NEW backbone every call;
         build one per region. A frozen, shared backbone would contradict
         "learns ... independently".
* "Both models will use the identical EfficientNetB0 backbone" (Appendix 1)
      -> identical architecture and identical ImageNet starting weights as the
         Baseline. Use the SAME unfreezing depth as the Baseline's
         unfreeze_for_finetuning() when fine-tuning (see unfreeze_top_layers).

Training on CPU (two stages, per region):
  Stage 1 -- backbone frozen: extract_features() once, cache to .npy, train
             only the Dense head on the cached vectors (fast).
  Stage 2 -- fine-tune: unfreeze_top_layers(), train the region's end-to-end
             model (hybrid_classifier.build_region_model) at a low learning
             rate. This is where the CNN "learns" from the region.

Full Face is NOT a separate CNN input. Appendix 3: "the Full Face setup formed
by combining the predictions of all five regions" -- see
hybrid_classifier.combine_region_predictions().

Usage (run from ml_pipeline/):
    python -m src.hueview.cnn_features --smoke            # no dataset, no download
    python -m src.hueview.cnn_features --smoke --imagenet # also checks real weights
"""
from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np

from .regions import MIN_VALID_PIXELS, REGION_ORDER

# Kept as REGIONS for readability in this module; always identical to
# regions.REGION_ORDER (the shared contract for Phases 7.3 - 9).
REGIONS = REGION_ORDER

INPUT_SIZE = 224          # manuscript: faces resized to 224 x 224
CNN_DIM = 1280            # EfficientNetB0 global-average-pooled output
CNN_OUTPUT_DIM = CNN_DIM  # alias used by older modules
LAB_DIM = 3               # Phase 8.2: [L*, a*, b*] per region
FUSED_DIM = CNN_DIM + LAB_DIM


# ---------------------------------------------------------------------------
# Patch preparation
# ---------------------------------------------------------------------------
def prepare_patch(
    rgb: np.ndarray,
    mask: np.ndarray,
    size: int = INPUT_SIZE,
    fill: str = "zero",
    crop: bool = False,
    min_valid: int = MIN_VALID_PIXELS,
) -> Optional[np.ndarray]:
    """
    Turn one region's SSR image + skin mask into an EfficientNetB0 input.

    Args:
        rgb:  HxWx3 uint8 RGB -- the SSR-normalized face (RegionPatch.image).
              cv2.imread returns BGR; convert with cv2.cvtColor first.
        mask: HxW, nonzero = valid skin pixel (RegionPatch.skin_mask).
        size: output side length (224).
        fill: "zero" (manuscript, Stage 3). "mean" = experiment only.
        crop: False (manuscript + regions.py contract): keep the full frame.
              True = experiment only: crop to the mask's bounding box.
        min_valid: return None below this many skin pixels (regions.py).

    Returns:
        float32 (size, size, 3), values 0..255 (Keras' EfficientNetB0 rescales
        internally -- do NOT divide by 255), or None if too few skin pixels.
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
        patch = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        pmask = valid[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    if fill == "zero":
        fill_color = np.zeros(3, np.float32)
    elif fill == "mean":
        fill_color = patch[pmask].mean(axis=0)
    else:
        raise ValueError(f"fill must be 'zero' or 'mean', got {fill!r}")
    patch = patch.copy()
    patch[~pmask] = fill_color

    h, w = patch.shape[:2]
    if (h, w) == (size, size):
        return patch

    # Only reached when crop=True or the input is not 224x224: letterbox,
    # keeping aspect ratio so skin texture is not stretched.
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
# Backbone -- one per region
# ---------------------------------------------------------------------------
def build_backbone(region: str = "",
                   weights: Optional[str] = "imagenet",
                   trainable: bool = False,
                   input_size: int = INPUT_SIZE):
    """
    A NEW EfficientNetB0 feature extractor: (224,224,3) -> (1280,).

    Call once PER REGION: every call returns fresh weights (ImageNet
    initialisation), so each region's CNN learns independently.
    Starts frozen (stage 1); use unfreeze_top_layers() for stage 2.
    """
    from tensorflow import keras

    if region and region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}. Options: {list(REGIONS)}")
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
        pooling="avg",
    )
    base.name = f"efficientnetb0_{region}" if region else "efficientnetb0"
    base.trainable = trainable
    return base


def unfreeze_top_layers(backbone, from_layer: str = "block6a_expand_conv") -> int:
    """
    Stage 2 fine-tuning: unfreeze every layer from `from_layer` to the top,
    keeping BatchNormalization layers frozen (standard for small-batch
    fine-tuning; BN also runs in inference mode because the models call the
    backbone with training=False).

    Set `from_layer` to the SAME depth the Baseline unfreezes in
    src/baseline/model.py::unfreeze_for_finetuning, so both models' CNNs are
    trained identically (Appendix 1). Recompile the model afterwards.

    Returns the number of trainable weights.
    """
    from tensorflow import keras

    names = [layer.name for layer in backbone.layers]
    if from_layer not in names:
        raise ValueError(f"{from_layer!r} is not a layer of EfficientNetB0")
    start = names.index(from_layer)

    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= start
                           and not isinstance(layer, keras.layers.BatchNormalization))
    return int(sum(np.prod(w.shape) for w in backbone.trainable_weights))


def extract_features(backbone, patches: np.ndarray,
                     batch_size: int = 32) -> np.ndarray:
    """
    Stage 1: run a FROZEN backbone over prepared patches -> (N, 1280) float32.
    Cache the result to .npy and train the Dense head on it.
    """
    return backbone.predict(patches, batch_size=batch_size, verbose=0).astype(
        np.float32)


# ---------------------------------------------------------------------------
# Smoke test (no dataset needed)
# ---------------------------------------------------------------------------
def _smoke(use_imagenet: bool) -> int:
    import tensorflow as tf
    from tensorflow import keras

    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [ok]   {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name} {detail}")

    print("\n== prepare_patch (Stage 3 / Stage 4 input) ==")
    rng = np.random.default_rng(0)
    img = rng.integers(1, 256, (224, 224, 3), dtype=np.uint8)
    mask = np.zeros((224, 224), np.uint8)
    mask[40:100, 60:180] = 1
    mask[60:70, 100:120] = 0            # hole left by the HSV filter
    inside, outside = mask.astype(bool), ~mask.astype(bool)

    p = prepare_patch(img, mask)
    check("output (224,224,3) float32",
          p is not None and p.shape == (224, 224, 3) and p.dtype == np.float32)
    check("default = full 224 frame, skin pixels unchanged (no crop)",
          p is not None and np.allclose(p[inside], img[inside]))
    check("default = everything outside the skin mask is zero (Stage 3)",
          p is not None and np.allclose(p[outside], 0))
    check("values stay 0..255 (no /255)", p is not None and p.max() <= 255)
    pc = prepare_patch(img, mask, crop=True)
    check("crop=True still available as an experiment",
          pc is not None and pc.shape == (224, 224, 3))
    check("empty mask -> None", prepare_patch(img, np.zeros_like(mask)) is None)
    tiny = np.zeros_like(mask); tiny[:3, :3] = 1
    check("below MIN_VALID_PIXELS -> None", prepare_patch(img, tiny) is None)
    try:
        prepare_patch(img, mask[:10]); check("mismatched mask raises", False)
    except ValueError:
        check("mismatched mask raises", True)

    print("\n== one EfficientNetB0 per region ==")
    w = "imagenet" if use_imagenet else None
    backbones = {r: build_backbone(r, weights=w) for r in REGIONS}
    check("5 backbones, one per region", len(backbones) == 5)
    check("each region has its OWN backbone object",
          len({id(b) for b in backbones.values()}) == 5)
    check("output = 1280-d vector",
          all(tuple(b.output.shape) == (None, CNN_DIM) for b in backbones.values()))
    check("regions share the same architecture",
          len({b.count_params() for b in backbones.values()}) == 1)

    bf, bj = backbones["forehead"], backbones["jawline"]
    check("stage 1: frozen (0 trainable weights)", len(bf.trainable_weights) == 0)
    feats = extract_features(bf, np.stack([p, p]))
    check("extract_features -> (2,1280) float32, finite",
          feats.shape == (2, CNN_DIM) and feats.dtype == np.float32
          and np.isfinite(feats).all())

    n_train = unfreeze_top_layers(bf)
    bn_frozen = all(not l.trainable for l in bf.layers
                    if isinstance(l, keras.layers.BatchNormalization))
    check("stage 2: top layers trainable", n_train > 0, str(n_train))
    check("stage 2: BatchNorm stays frozen", bn_frozen)
    check("stage 2: early layers stay frozen",
          not bf.get_layer("stem_conv").trainable)
    check("unfreezing forehead does NOT touch jawline's CNN (independent)",
          len(bj.trainable_weights) == 0)
    check("FUSED_DIM per region = 1283", FUSED_DIM == 1283)

    print(f"\n  backbone params per region: {bf.count_params():,}"
          f"  (trainable in stage 2: {n_train:,})")
    print(f"\n{passed} passed, {failed} failed  (TF {tf.__version__}, "
          f"weights={'imagenet' if use_imagenet else 'random'})")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 8.1 CNN feature branch")
    ap.add_argument("--smoke", action="store_true",
                    help="run checks without any dataset")
    ap.add_argument("--imagenet", action="store_true",
                    help="use real ImageNet weights (downloads ~16 MB once)")
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(_smoke(args.imagenet))
    ap.print_help()
