"""
Phase 8.3 -- Feature Fusion & Classification
Manuscript: Chapter 3, "Fused Classification & Output"; Data Generation
(Feature Selection, third step); Appendix 1 (Training and Validation);
Appendix 3 (six region setups).

-------------------------------------------------------------------------------
What the manuscript specifies, and how this file follows it
-------------------------------------------------------------------------------
* "For each facial region, the CNN feature vector is combined with the
  corresponding CIELAB feature vector through a feature concatenation layer"
      -> Concatenate([CNN 1280 (8.1), CIELAB 3 (8.2)]) = 1283, per region.
* "The fused representation is then passed through a fully connected layer
  to produce a skin tone classification score for each region ... region-wise
  skin tone predictions independently"
      -> ONE classifier per region: Dense(6, softmax) over SCC-1..SCC-6, and
         (from 8.1) one EfficientNetB0 per region. Five independent models.
* "the Full Face setup formed by combining the predictions of all five
  regions" (Appendix 3)
      -> Full Face has NO model of its own. combine_region_predictions()
         combines the five regional predictions (majority vote; see below).
* Compiled with Adam + categorical cross-entropy (build plan, Phase 8.3).
  STAGE1_LR / STAGE2_LR are starting values; Appendix 1 tunes learning rate,
  batch size and epochs on the validation set during training.

Training itself (Macro F1 checkpointing, early stopping, augmentation --
Appendix 1) is NOT Phase 8; it lives in the training phase.

NOT IN THE MANUSCRIPT (add to Chapter 3 or Appendix 1):
  * optimizer and loss -- Adam + categorical cross-entropy come from the
    build plan ("compile identically to baseline").
  * how the Full Face predictions are combined -- majority vote is used,
    the same rule the manuscript already uses to combine the five regional
    undertone labels (Stage 6); ties go to the class with the higher mean
    probability. method="mean_prob" is available if the adviser prefers.

Training per region (see cnn_features.py):
  Stage 1 -- frozen CNN: cache extract_features(), fit the head (fast on CPU).
  Stage 2 -- fine-tune: unfreeze_top_layers(), wrap with build_region_model(),
             recompile at STAGE2_LR, fit on images (training phase).

Usage (run from ml_pipeline/):
    python -m src.hueview.hybrid_classifier --smoke
"""
from __future__ import annotations

import argparse
from typing import Dict, Mapping, Sequence

import numpy as np

from .cnn_features import CNN_DIM, INPUT_SIZE, LAB_DIM
from .region_selector import FULL_FACE, ConfigurationOutput, assert_feature_shapes
from .regions import REGION_ORDER

NUM_CLASSES = 6                                    # SCC-1 .. SCC-6
SCC_LABELS = tuple(f"SCC-{i}" for i in range(1, NUM_CLASSES + 1))
STAGE1_LR = 1e-3                                   # tune on validation set
STAGE2_LR = 1e-5                                   # tune on validation set


# ---------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------
def _check_region(region: str) -> None:
    if region == FULL_FACE:
        raise ValueError(
            "Full Face has no classifier of its own (Appendix 3: 'combining the "
            "predictions of all five regions'). Use combine_region_predictions().")
    if region not in REGION_ORDER:
        raise ValueError(f"Unknown region {region!r}. Options: {list(REGION_ORDER)}")


def compile_model(model, learning_rate: float = STAGE1_LR):
    """Adam + categorical cross-entropy (build plan). Recompile after unfreezing."""
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_fusion_head(region: str,
                      num_classes: int = NUM_CLASSES,
                      cnn_dim: int = CNN_DIM,
                      learning_rate: float = STAGE1_LR):
    """
    One region's fusion + classification head:
        [CNN (1280), CIELAB (3)] -> Concatenate -> Dense(6, softmax)
    Compiled, ready for stage 1 training on cached CNN vectors.
    """
    from tensorflow import keras

    _check_region(region)
    cnn_in = keras.Input((cnn_dim,), name="cnn_features")
    lab_in = keras.Input((LAB_DIM,), name="cielab_features")
    fused = keras.layers.Concatenate(name="feature_fusion")([cnn_in, lab_in])
    out = keras.layers.Dense(num_classes, activation="softmax", name="scc_output")(fused)
    model = keras.Model([cnn_in, lab_in], out, name=f"hueview_head_{region}")
    return compile_model(model, learning_rate)


def build_hueview_heads(regions: Sequence[str] = REGION_ORDER, **kwargs) -> Dict[str, object]:
    """Five independent heads, one per region (no Full Face head)."""
    return {r: build_fusion_head(r, **kwargs) for r in regions}


def build_region_model(region: str, backbone, head,
                       input_size: int = INPUT_SIZE):
    """
    One region's complete HueView classifier:
        [patch (224,224,3), CIELAB (3)] -> that region's EfficientNetB0
                                        -> that region's head -> softmax(6)

    Reuses the given layer objects, so a head trained in stage 1 carries over.
    Use it for stage 2 fine-tuning (after cnn_features.unfreeze_top_layers
    and compile_model(..., STAGE2_LR)) and for inference.
    """
    from tensorflow import keras

    _check_region(region)
    img_in = keras.Input((input_size, input_size, 3), name=f"{region}_patch")
    lab_in = keras.Input((LAB_DIM,), name="cielab_features")
    feat = backbone(img_in, training=False)   # BatchNorm in inference mode
    out = head([feat, lab_in])
    return keras.Model([img_in, lab_in], out, name=f"hueview_{region}")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_configuration(cfg: ConfigurationOutput,
                          cnn_vector: np.ndarray,
                          cielab_vector: np.ndarray,
                          heads: Mapping[str, object],
                          cnn_dim: int = CNN_DIM) -> np.ndarray:
    """
    One region's (6,) softmax from its cached CNN vector + CIELAB triple.
    Runs the Phase 8 shape checkpoint (region_selector.assert_feature_shapes)
    first, so a mismatched vector raises instead of training on garbage.
    """
    _check_region(cfg.config)
    assert_feature_shapes(cfg, cnn_vector, cielab_vector, cnn_dim=cnn_dim)
    probs = heads[cfg.config].predict(
        [cnn_vector[np.newaxis, :], cielab_vector[np.newaxis, :]], verbose=0)
    return probs[0]


def combine_region_predictions(region_probs: Mapping[str, np.ndarray],
                               method: str = "vote") -> Dict[str, object]:
    """
    Full Face = combination of the regional predictions (Appendix 3).

    Args:
        region_probs: {region: (6,) softmax} for the regions that were usable
                      for this image (missing regions are simply absent).
        method: "vote"      -- majority vote of regional labels (default; same
                               rule as the Stage 6 undertone vote); ties go to
                               the class with the higher mean probability.
                "mean_prob" -- argmax of the averaged probabilities.

    Returns {"scc_index": 0-5, "label": "SCC-n", "probs": mean (6,),
             "n_regions": int}.
    """
    if not region_probs:
        raise ValueError("No regional predictions to combine.")
    unknown = [r for r in region_probs if r not in REGION_ORDER]
    if unknown:
        raise ValueError(f"Not regions: {unknown}")

    stack = np.stack([np.asarray(p, float) for p in region_probs.values()])
    mean_p = stack.mean(axis=0)
    if method == "mean_prob":
        idx = int(mean_p.argmax())
    elif method == "vote":
        votes = np.bincount(stack.argmax(axis=1), minlength=stack.shape[1])
        tied = np.flatnonzero(votes == votes.max())
        idx = int(tied[mean_p[tied].argmax()])
    else:
        raise ValueError(f"method must be 'vote' or 'mean_prob', got {method!r}")
    return {"scc_index": idx, "label": SCC_LABELS[idx], "probs": mean_p,
            "n_regions": len(region_probs)}


# ---------------------------------------------------------------------------
# Smoke test (no dataset needed)
# ---------------------------------------------------------------------------
def _smoke() -> int:
    import tensorflow as tf
    from tensorflow import keras

    from .cielab_features import build_cielab_vector
    from .cnn_features import (build_backbone, extract_features,
                               prepare_patch, unfreeze_top_layers)
    from .region_selector import RegionalConfigurationSelector
    from .regions import STATUS_OK, RegionPatch

    keras.utils.set_random_seed(0)
    passed = failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [ok]   {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name} {detail}")

    print("\n== heads: one per region, Full Face has none ==")
    heads = build_hueview_heads()
    check("5 heads, one per region", list(heads) == list(REGION_ORDER), str(list(heads)))
    try:
        build_fusion_head(FULL_FACE); check("no Full Face head can be built", False)
    except ValueError:
        check("no Full Face head can be built", True)
    widths = {r: int(h.get_layer("feature_fusion").output.shape[-1]) for r, h in heads.items()}
    check("checkpoint: each region fuses 1280 + 3 = 1283",
          all(v == 1283 for v in widths.values()), str(widths))
    check("checkpoint: Dense input width = 1283",
          all(h.get_layer("scc_output").kernel.shape[0] == 1283 for h in heads.values()))
    check("fully connected layer -> 6-unit softmax",
          all(h.get_layer("scc_output").units == 6
              and h.get_layer("scc_output").activation.__name__ == "softmax"
              for h in heads.values()))
    check("heads independent (no shared weights)",
          heads["forehead"].get_layer("scc_output") is not heads["jawline"].get_layer("scc_output"))
    h0 = heads["forehead"]
    check("compiled: Adam + categorical cross-entropy",
          isinstance(h0.optimizer, keras.optimizers.Adam)
          and "categorical_crossentropy" in str(h0.loss))

    print("\n== head trains on cached CNN vectors (stage 1) ==")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 96)
    Xc = rng.normal(size=(96, CNN_DIM)).astype("float32")
    Xc[np.arange(96), y] += 3.0
    Xl = rng.normal(size=(96, 3)).astype("float32")
    hist = h0.fit([Xc, Xl], keras.utils.to_categorical(y, 6), epochs=10,
                  batch_size=16, verbose=0)
    check("loss decreases", hist.history["loss"][-1] < hist.history["loss"][0])
    check("probabilities sum to 1",
          np.allclose(h0.predict([Xc[:4], Xl[:4]], verbose=0).sum(1), 1, atol=1e-5))

    print("\n== full pipeline for all 5 regions: 7.5 -> 8.1 -> 8.2 -> 8.3 ==")
    H = W = 224
    boxes = {"forehead": (10, 50, 50, 170), "left_cheek": (80, 140, 130, 190),
             "right_cheek": (80, 140, 30, 90), "nose_bridge": (80, 140, 95, 125),
             "jawline": (160, 210, 90, 135)}
    ssr = rng.integers(40, 230, (H, W, 3), dtype=np.uint8)
    patches = {}
    for r, (y0, y1, x0, x1) in boxes.items():
        m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
        img = np.zeros_like(ssr); img[m] = ssr[m]
        patches[r] = RegionPatch(r, img, m, m.copy(), STATUS_OK, int(m.sum()), int(m.sum()))
    outs = RegionalConfigurationSelector(configurations=REGION_ORDER).route_all(patches, "synthetic")
    backbones = {r: build_backbone(r, weights=None) for r in REGION_ORDER}

    region_probs = {}
    for r, cfg in outs.items():
        x = prepare_patch(cfg.image, cfg.mask)
        cnn_vec = extract_features(backbones[r], x[None])[0]
        lab_vec = build_cielab_vector(cfg).astype("float32")
        region_probs[r] = predict_configuration(cfg, cnn_vec, lab_vec, heads)
    check("each region -> its own (6,) softmax",
          len(region_probs) == 5 and all(p.shape == (6,) and np.isclose(p.sum(), 1, atol=1e-5)
                                         for p in region_probs.values()))
    ff = combine_region_predictions(region_probs)
    check("Full Face = combination of the 5 regional predictions",
          ff["n_regions"] == 5 and ff["label"] in SCC_LABELS)
    try:
        predict_configuration(outs["forehead"], cnn_vec, np.zeros(15, "float32"), heads)
        check("15-d CIELAB into a region head is rejected", False)
    except ValueError:
        check("15-d CIELAB into a region head is rejected", True)

    print("\n== region model + stage 2 fine-tuning ==")
    r = "forehead"
    cfg = outs[r]
    x = prepare_patch(cfg.image, cfg.mask)[None]
    lab = build_cielab_vector(cfg)[None].astype("float32")
    model = build_region_model(r, backbones[r], heads[r])
    two_step = heads[r].predict([extract_features(backbones[r], x), lab], verbose=0)
    check("region model = its CNN + its head (identical output)",
          np.allclose(model.predict([x, lab], verbose=0), two_step, atol=1e-5))

    unfreeze_top_layers(backbones[r])
    compile_model(model, STAGE2_LR)
    before_f = [w.numpy().copy() for w in backbones[r].trainable_weights]
    before_j = [w.numpy().copy() for w in backbones["jawline"].weights]
    model.fit([np.repeat(x, 4, 0), np.repeat(lab, 4, 0)],
              keras.utils.to_categorical([2, 2, 2, 2], 6), epochs=1, verbose=0)
    check("stage 2 updates this region's CNN (it 'learns')",
          any(not np.allclose(a, b.numpy()) for a, b in zip(before_f, backbones[r].trainable_weights)))
    check("other regions' CNNs untouched (independent)",
          all(np.allclose(a, b.numpy()) for a, b in zip(before_j, backbones["jawline"].weights)))

    print("\n== Full Face combination rules ==")
    oh = lambda i, s=0.6: np.eye(6)[i] * s + (1 - s) / 6
    rp = {"forehead": oh(2), "left_cheek": oh(2), "right_cheek": oh(3),
          "nose_bridge": oh(3), "jawline": oh(2)}
    check("majority vote: 3 x SCC-3 beats 2 x SCC-4",
          combine_region_predictions(rp)["label"] == "SCC-3")
    check("tie -> higher mean probability",
          combine_region_predictions({"forehead": oh(1, .9), "jawline": oh(4, .5)})["label"] == "SCC-2")
    check("mean_prob option", combine_region_predictions(rp, "mean_prob")["label"] == "SCC-3")
    check("works with fewer than 5 usable regions",
          combine_region_predictions({k: rp[k] for k in list(rp)[:3]})["n_regions"] == 3)

    print(f"\n{passed} passed, {failed} failed  (TF {tf.__version__})")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 8.3 fusion + classification")
    ap.add_argument("--smoke", action="store_true", help="run checks without any dataset")
    if ap.parse_args().smoke:
        raise SystemExit(_smoke())
    ap.print_help()
