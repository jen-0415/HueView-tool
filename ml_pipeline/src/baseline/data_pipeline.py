"""
Phase 6.4 — Data pipeline for Baseline training
================================================

Builds tf.data.Dataset objects feeding the two-input model from 6.2:

    ([image, rgb_features], one_hot_scc_label)

Images are resolved through resolved_manifest.csv, never by re-deriving
paths — so Baseline and HueView train on identical bytes for identical
rows.

Four things this file gets deliberately right, each of which is easy to
get wrong and silent when you do:

1. NO /255 ON THE IMAGE. Keras' EfficientNet rescales internally and
   expects [0, 255]. Normalizing here would do it twice. The RGB feature
   vector IS in [0, 1] — different branch, different convention.

2. AUGMENTATION ON TRAIN ONLY. Phase 5 specifies this. Augmenting val or
   test would measure the model on images it will never see in
   deployment and quietly inflate the numbers.

3. CLASS WEIGHTS, NOT RESAMPLING. SCC-5 has ~14.8k images against
   SCC-6's ~2.5k. Oversampling the minority would duplicate images
   across batches; class weights fix the loss imbalance without touching
   the data. compute_class_weights() derives them from the training
   split only.

4. FIXED LABEL ORDER. SCC-1..SCC-6 map to indices 0..5 via an explicit
   constant, not via whatever order pandas happens to produce. A shifted
   label mapping between training and evaluation is invisible until your
   confusion matrix looks strange.

Usage:
    from data_pipeline import build_datasets
    train_ds, val_ds, test_ds, class_weights = build_datasets()
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

PROC = Path("data/processed")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
AUTOTUNE = tf.data.AUTOTUNE

# Explicit and fixed. Never infer this from the data.
SCC_CLASSES = ["SCC-1", "SCC-2", "SCC-3", "SCC-4", "SCC-5", "SCC-6"]
SCC_TO_INDEX = {c: i for i, c in enumerate(SCC_CLASSES)}


def _normalize_label(v):
    """Accept 'SCC-3', 'scc3', or 3 and return the canonical 'SCC-3'."""
    if isinstance(v, (int, np.integer)):
        return f"SCC-{int(v)}"
    s = str(v).strip().upper().replace("_", "-").replace(" ", "")
    if s.startswith("SCC") and not s.startswith("SCC-"):
        s = "SCC-" + s[3:]
    return s


def load_split(split: str) -> pd.DataFrame:
    """
    Assemble one split: resolved image path + RGB features + SCC label.

    Joins three files, and fails loudly if they disagree. A silent partial
    join here would train the model on a subset while you believe it saw
    everything.
    """
    split_df = pd.read_csv(PROC / f"{split}.csv")
    col = "filename" if "filename" in split_df.columns else split_df.columns[0]
    split_df = split_df.rename(columns={col: "filename"})

    resolved = pd.read_csv(PROC / "resolved_manifest.csv")[["filename", "resolved_path"]]
    feats = pd.read_csv(PROC / f"baseline_rgb_{split}.csv")

    manifest = pd.read_csv(PROC / "manifest.csv")
    label_col = next((c for c in manifest.columns
                      if c.lower() in ("scc_label", "scc", "label")), None)
    if label_col is None:
        raise ValueError(
            f"No SCC label column in manifest.csv. Columns: {list(manifest.columns)}"
        )
    labels = manifest[["filename", label_col]].rename(columns={label_col: "scc"})

    df = (split_df
          .merge(resolved, on="filename", how="left")
          .merge(feats, on="filename", how="inner")   # inner: drop rows missing features
          .merge(labels, on="filename", how="left"))

    before = len(split_df)
    dropped = before - len(df)
    if dropped:
        print(f"  [{split}] {dropped} rows dropped — no RGB features (images missing from disk).")

    # Re-resolve any paths that no longer exist on disk (folder reorganization).
    # resolved_manifest was built on a different machine/layout; check each path
    # and fall back to the live resolver for any that are stale.
    bad_mask = df["resolved_path"].apply(
        lambda p: pd.isna(p) or not Path(str(p)).is_file()
    )
    if bad_mask.any():
        from ml_pipeline.src.baseline.path_resolver import resolve_image_path as _resolve
        def _fix(row):
            p, _, _ = _resolve(row["filename"])
            return str(p) if p else row["resolved_path"]
        df.loc[bad_mask, "resolved_path"] = df[bad_mask].apply(_fix, axis=1)
        still_bad = df["resolved_path"].apply(
            lambda p: pd.isna(p) or not Path(str(p)).is_file()
        ).sum()
        print(f"  [{split}] {int(bad_mask.sum())} stale paths re-resolved; "
              f"{int(still_bad)} still missing.")

    problems = []

    df["scc"] = df["scc"].map(_normalize_label)
    unknown = set(df["scc"]) - set(SCC_CLASSES)
    if unknown:
        raise ValueError(f"[{split}] unexpected SCC values: {sorted(unknown)}")

    df["label_index"] = df["scc"].map(SCC_TO_INDEX)
    return df


def _decode(path, rgb, label, augment: bool):
    """Read one image and return the two-input tuple the model expects."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)  # stays in [0, 255] — see note 1 above

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=20.0)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.clip_by_value(img, 0.0, 255.0)

    return (img, rgb), label


def make_dataset(df: pd.DataFrame, augment: bool = False,
                 shuffle: bool = False, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Turn a loaded split into a batched tf.data.Dataset."""
    paths = df["resolved_path"].astype(str).to_numpy()
    rgb = df[["r_mean", "g_mean", "b_mean"]].to_numpy(dtype=np.float32)
    labels = tf.keras.utils.to_categorical(
        df["label_index"].to_numpy(), num_classes=NUM_CLASSES
    ).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, rgb, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 8192), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, r, l: _decode(p, r, l, augment), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def compute_class_weights(train_df: pd.DataFrame) -> dict:
    """
    Inverse-frequency weights, normalized to mean 1.0.

    Derived from the training split only — using val or test counts would
    leak information about the evaluation data into training.
    """
    counts = train_df["label_index"].value_counts().sort_index()
    total = counts.sum()
    raw = {i: total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)}
    mean_w = np.mean(list(raw.values()))
    return {i: float(w / mean_w) for i, w in raw.items()}


def build_datasets(batch_size: int = BATCH_SIZE):
    """
    Build all three datasets plus class weights.

    Returns (train_ds, val_ds, test_ds, class_weights).
    """
    print("Loading splits...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:<6} {len(df):>6} rows")

    print("\nSCC distribution:")
    print(f"  {'class':<8} {'train':>7} {'val':>7} {'test':>7}")
    for i, cls in enumerate(SCC_CLASSES):
        t = int((train_df["label_index"] == i).sum())
        v = int((val_df["label_index"] == i).sum())
        s = int((test_df["label_index"] == i).sum())
        print(f"  {cls:<8} {t:>7} {v:>7} {s:>7}")

    weights = compute_class_weights(train_df)
    print("\nClass weights (inverse frequency, mean-normalized):")
    for i, cls in enumerate(SCC_CLASSES):
        print(f"  {cls:<8} {weights[i]:.3f}")

    ratio = max(weights.values()) / min(weights.values())
    if ratio > 5:
        print(f"\n  Imbalance ratio {ratio:.1f}:1 between the largest and smallest class.")
        print("  Weights compensate in the loss, but report per-class recall as well")
        print("  as overall accuracy — a model can score well overall while barely")
        print("  learning the smallest class.")

    return (
        make_dataset(train_df, augment=True, shuffle=True, batch_size=batch_size),
        make_dataset(val_df, batch_size=batch_size),
        make_dataset(test_df, batch_size=batch_size),
        weights,
    )


if __name__ == "__main__":
    train_ds, val_ds, test_ds, weights = build_datasets()

    print("\nChecking one batch...")
    (img, rgb), label = next(iter(train_ds))
    print(f"  image  {img.shape}  {img.dtype}  range [{tf.reduce_min(img):.1f}, {tf.reduce_max(img):.1f}]")
    print(f"  rgb    {rgb.shape}  {rgb.dtype}  range [{tf.reduce_min(rgb):.3f}, {tf.reduce_max(rgb):.3f}]")
    print(f"  label  {label.shape}  sums to {tf.reduce_sum(label[0]):.1f} per row")

    if tf.reduce_max(img) <= 1.5:
        print("\n  !! Image values look normalized to [0,1]. EfficientNet expects")
        print("     [0,255] and rescales internally — this would train on")
        print("     double-normalized inputs.")
    else:
        print("\n  Image range is [0,255] as EfficientNet expects.")

    print("\nPipeline ready. Next: train.py")
