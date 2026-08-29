"""
Phase 6.2 — Baseline Skin Tone Classification Branch
====================================================

Builds the Baseline model:

    224x224x3 raw RGB image ──> EfficientNetB0 (include_top=False)
                                        │ global average pool
                                        ▼
                                 1280-d CNN vector
                                        │
    [r_mean, g_mean, b_mean] ───────────┤ concatenate
    (from Phase 6.1)                    ▼
                                 Dense(6, softmax)
                                        ▼
                                  SCC-1 ... SCC-6

The Baseline is a CONTROL condition. It deliberately omits everything
HueView adds — no SSR, no regional segmentation, no CIELAB. Whatever it
scores is the number HueView must beat, so nothing "helpful" should be
added here that HueView doesn't also get. Keeping it plain is the point.

Two things worth knowing before you train:

1. NO MANUAL NORMALIZATION. Keras' EfficientNet expects pixels in the
   [0, 255] range and rescales internally as part of the model graph.
   Dividing by 255 first would normalize twice and quietly degrade
   accuracy. This matches the build plan's "raw, un-normalized" wording.

2. THE 3 RGB FEATURES ARE HEAVILY OUTNUMBERED (3 vs 1280). Straight
   concatenation is what the methodology specifies, so that's the default
   here, but the CNN branch will dominate the gradient almost entirely.
   `rgb_projection_dim` is provided if you later want to widen the RGB
   branch — leave it None to match the manuscript exactly. If you do
   change it, HueView's fusion needs the same treatment or the comparison
   stops being like-for-like.

Run directly to print the architecture:
    python src/baseline/model.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 6          # SCC-1 .. SCC-6
IMAGE_SIZE = (224, 224)
RGB_FEATURE_DIM = 3      # [r_mean, g_mean, b_mean] from Phase 6.1


def build_baseline_model(
    num_classes: int = NUM_CLASSES,
    image_size=IMAGE_SIZE,
    rgb_feature_dim: int = RGB_FEATURE_DIM,
    freeze_backbone: bool = True,
    rgb_projection_dim=None,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build and compile the Baseline classifier.

    Args:
        num_classes:     output classes (6 for SCC-1..SCC-6)
        image_size:      (H, W) of the input images
        rgb_feature_dim: length of the Phase 6.1 feature vector
        freeze_backbone: True trains only the head (fast, stable, the usual
                         starting point). Set False to fine-tune the whole
                         network — do that as a second pass with a much
                         lower learning rate, e.g. 1e-5.
        rgb_projection_dim: None matches the manuscript (raw concatenation).
                         An int inserts a Dense layer on the RGB branch so
                         it isn't swamped by the 1280-d CNN vector.
        dropout:         dropout before the output layer
        learning_rate:   Adam learning rate

    Returns:
        A compiled keras.Model taking [image, rgb_features].
    """
    image_input = keras.Input(shape=(*image_size, 3), name="image")
    rgb_input = keras.Input(shape=(rgb_feature_dim,), name="rgb_features")

    # --- CNN branch -------------------------------------------------
    # weights="imagenet" gives a pretrained starting point. include_top=False
    # drops ImageNet's 1000-class head; pooling="avg" collapses the feature
    # map to a single 1280-d vector.
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
        pooling="avg",
    )
    # With input_tensor=, Keras inlines the backbone's layers into this graph
    # rather than nesting a sub-model, so freeze them individually.
    if freeze_backbone:
        for layer in backbone.layers:
            layer.trainable = False
    cnn_features = backbone.output  # (None, 1280)

    # --- Global RGB branch (Phase 6.1) ------------------------------
    if rgb_projection_dim:
        rgb_branch = layers.Dense(
            rgb_projection_dim, activation="relu", name="rgb_projection"
        )(rgb_input)
    else:
        rgb_branch = rgb_input

    # --- Fusion + classifier ----------------------------------------
    fused = layers.Concatenate(name="fusion")([cnn_features, rgb_branch])
    if dropout:
        fused = layers.Dropout(dropout, name="dropout")(fused)
    output = layers.Dense(num_classes, activation="softmax", name="scc_output")(fused)

    model = keras.Model(
        inputs=[image_input, rgb_input], outputs=output, name="baseline_effnet"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_finetuning(model: keras.Model, learning_rate: float = 1e-5) -> keras.Model:
    """
    Unfreeze the backbone for a second training pass.

    Because build_baseline_model passes `input_tensor=`, Keras inlines the
    EfficientNet layers into this model rather than nesting a sub-model.
    So every layer is walked directly here — looking for a nested
    keras.Model would silently match nothing and leave everything frozen.

    BatchNormalization layers stay frozen: updating their running statistics
    on small batches destabilizes fine-tuning, which is standard practice
    for transfer learning.

    Always recompile after changing `trainable`, or Keras keeps the old
    graph and the change has no effect.
    """
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    print("Building Baseline model (Phase 6.2)...\n")
    model = build_baseline_model()

    total = model.count_params()
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))

    print(f"Inputs:  {[i.name for i in model.inputs]}")
    print(f"Output:  {model.output_shape}  (softmax over {NUM_CLASSES} SCC classes)")
    print(f"\nParameters: {total:,} total, {trainable:,} trainable")
    print("(Backbone frozen — only the classifier head trains on the first pass.)")

    # Smoke test: does a forward pass actually run?
    dummy_img = np.random.randint(0, 256, size=(2, 224, 224, 3)).astype("float32")
    dummy_rgb = np.random.rand(2, 3).astype("float32")
    preds = model.predict([dummy_img, dummy_rgb], verbose=0)

    print(f"\nForward pass OK. Output shape {preds.shape}")
    print(f"Row sums (softmax should give 1.0): {preds.sum(axis=1)}")
    print("\nNext: 6.3 (rule-based undertone) and 6.4 (training loop).")
