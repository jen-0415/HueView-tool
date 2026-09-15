"""
Phase 8.3 -- Feature Fusion & Classification

"For each facial region, the CNN feature vector is combined with the
corresponding CIELAB feature vector through a feature concatenation layer...
The fused representation is then passed through a fully connected layer to
produce a skin tone classification score for each region."

TWO HEADS, ONE SHARED BACKBONE -- why this isn't a contradiction:

The manuscript's "identical EfficientNetB0 backbone" requirement (see
cnn_features.py) and its explicit "3 values x 5 regions" vs. single-region
"3 values" CIELAB widths are both real requirements, but a single Dense
layer needs a FIXED input width -- it cannot accept both a 3-dim and a
15-dim CIELAB vector. So "shared" applies to the CNN feature EXTRACTOR
(1280-dim output, always, built separately in cnn_features.py) -- not to
the fusion/classification layer, which necessarily splits into:

    region_head:     fused width 1280 + 3  = 1283, shared across the 5
                      individual regions (forehead, left_cheek,
                      right_cheek, nose_bridge, jawline)
    full_face_head:   fused width 1280 + 15 = 1295, Full Face only

This is the minimal architecture that satisfies both stated requirements
simultaneously -- not a compromise, just what they jointly imply.

Compiled identically to the Baseline model (categorical cross-entropy,
Adam) per the manuscript's fairness requirement and the build plan's
explicit Phase 8.3 instruction to do so.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from .cnn_features import CNN_OUTPUT_DIM
from .region_selector import ConfigurationOutput, FULL_FACE, assert_feature_shapes

NUM_CLASSES = 6  # SCC-1 through SCC-6

REGION_HEAD = "region_head"
FULL_FACE_HEAD = "full_face_head"


def build_fusion_head(
    cielab_dim: int,
    num_classes: int = NUM_CLASSES,
    cnn_dim: int = CNN_OUTPUT_DIM,
    name: str = "hueview_head",
) -> tf.keras.Model:
    """
    One fusion + classification head: concatenate a CNN feature vector with
    a CIELAB feature vector, pass through a Dense(num_classes, softmax)
    layer. cielab_dim fixes this head's expected CIELAB width (3 or 15) --
    build one instance per width you need, not one instance reused for both.
    """
    cnn_input = tf.keras.Input(shape=(cnn_dim,), name="cnn_features")
    cielab_input = tf.keras.Input(shape=(cielab_dim,), name="cielab_features")
    fused = tf.keras.layers.Concatenate(name="feature_fusion")([cnn_input, cielab_input])
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="scc_output")(fused)

    model = tf.keras.Model(inputs=[cnn_input, cielab_input], outputs=output, name=name)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_hueview_heads(
    num_classes: int = NUM_CLASSES,
    cnn_dim: int = CNN_OUTPUT_DIM,
) -> Dict[str, tf.keras.Model]:
    """
    Both heads HueView needs: one shared across the 5 individual regions
    (CIELAB dim 3), one for Full Face (CIELAB dim 15). See module docstring
    for why two heads are necessary despite the "shared backbone" framing.
    """
    return {
        REGION_HEAD: build_fusion_head(cielab_dim=3, num_classes=num_classes, cnn_dim=cnn_dim, name=REGION_HEAD),
        FULL_FACE_HEAD: build_fusion_head(cielab_dim=15, num_classes=num_classes, cnn_dim=cnn_dim, name=FULL_FACE_HEAD),
    }


def head_for_config(config_name: str, heads: Dict[str, tf.keras.Model]) -> tf.keras.Model:
    """Route a ConfigurationOutput.config name to the head it belongs to."""
    return heads[FULL_FACE_HEAD] if config_name == FULL_FACE else heads[REGION_HEAD]


def predict_configuration(
    cfg: ConfigurationOutput,
    cnn_vector: np.ndarray,
    cielab_vector: np.ndarray,
    heads: Dict[str, tf.keras.Model],
    cnn_dim: int = CNN_OUTPUT_DIM,
) -> np.ndarray:
    """
    Run one ConfigurationOutput's fused features through the correct head.

    Calls region_selector.assert_feature_shapes() first -- the build plan's
    own Phase 8 checkpoint -- so a 15-d vector reaching the 3-d region_head
    (or vice versa) raises immediately instead of training silently on
    garbage.

    Returns:
        (num_classes,) softmax probabilities.
    """
    assert_feature_shapes(cfg, cnn_vector, cielab_vector, cnn_dim=cnn_dim)

    model = head_for_config(cfg.config, heads)
    probs = model.predict(
        [cnn_vector[np.newaxis, :], cielab_vector[np.newaxis, :]],
        verbose=0,
    )
    return probs[0]  # (num_classes,)
