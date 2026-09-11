"""
Phase 8.1 -- Shared EfficientNetB0 CNN Feature Branch

Per the manuscript: "the isolated, illumination-adjusted skin patches...
are fed sequentially -- one region at a time -- into the EfficientNetB0
backbone for deep feature extraction." Every mention of the CNN component
throughout the manuscript uses singular language ("the EfficientNetB0
backbone", never "backbones"), and the Research Instrument's fairness
requirement is explicit: "Both models will use the identical EfficientNetB0
backbone... to ensure that performance differences are attributable solely
to the input representation." So this is ONE shared backbone, fed each of
the six configurations' (5 regions + Full Face) masked image in turn --
not six separately-trained backbones.

Feeds the raw [0, 255] masked 224x224 RGB image directly, matching the
finding already recorded from the Baseline (Phase 6) work: Keras's
EfficientNetB0 (include_top=False) has its own internal Rescaling layer and
expects raw, un-normalized pixel values -- manually dividing by 255 first
would double-normalize.

Output is a 1280-dim vector via global average pooling (pooling="avg"),
matching the cnn_dim=1280 default already assumed by
region_selector.assert_feature_shapes().

Weights are ImageNet-pretrained by default, per the manuscript's literature
review framing this as standard transfer-learning practice for
task-specific data that's limited relative to ImageNet's scale. The first
call to build_cnn_backbone() downloads the pretrained weights (~29 MB) if
not already cached locally -- needs internet access once.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

CNN_OUTPUT_DIM = 1280
INPUT_SHAPE = (224, 224, 3)


def build_cnn_backbone(weights: str = "imagenet") -> tf.keras.Model:
    """
    The single shared EfficientNetB0 feature extractor used across all six
    configurations (5 regions + full_face). Not trained further here --
    Phase 10 decides whether/how to fine-tune; this just builds the model.
    """
    return EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=INPUT_SHAPE,
        pooling="avg",
    )


def extract_cnn_features(model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    """
    Args:
        model: from build_cnn_backbone().
        image: (224, 224, 3) uint8 RGB, raw [0, 255] -- e.g. a
               ConfigurationOutput's .image field. Do NOT divide by 255
               first.

    Returns:
        (1280,) float32 feature vector.
    """
    if tuple(image.shape) != INPUT_SHAPE:
        raise ValueError(f"Expected image shape {INPUT_SHAPE}, got {tuple(image.shape)}")

    batch = image.astype(np.float32)[np.newaxis, ...]  # (1, 224, 224, 3)
    features = model.predict(batch, verbose=0)
    return features[0]  # (1280,)


def extract_cnn_features_batch(model: tf.keras.Model, images: np.ndarray) -> np.ndarray:
    """
    Batched version -- feed a stack of (N, 224, 224, 3) raw RGB images at
    once. Much faster than calling extract_cnn_features in a loop; use this
    for full-dataset feature extraction during training data preparation.
    """
    if images.ndim != 4 or tuple(images.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"Expected (N, {INPUT_SHAPE}), got {images.shape}")
    return model.predict(images.astype(np.float32), verbose=0)
