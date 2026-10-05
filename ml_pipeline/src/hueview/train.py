"""
PHASE 10 — HueView training.  [MANUSCRIPT-ALIGNED REVISION v3]

Trains one fused (EfficientNetB0 + CIELAB) classifier per region configuration,
reusing the frozen Phase 5 split and the Baseline's training recipe.

Run order:
    python -m ml_pipeline.src.hueview.train --cache-lab    # once, per region
    python -m ml_pipeline.src.hueview.train --smoke        # 32-image sanity check
    python -m ml_pipeline.src.hueview.train --ablate-lab   # LAB branch matters?
    python -m ml_pipeline.src.hueview.train                # the real run

==========================================================================
WHAT v3 CHANGES, AND WHY IT MATTERS MORE THAN v2 DID
==========================================================================
v2 recomputed the HSV skin mask from the saved regional patch JPGs. That was
unnecessary and, as it turned out, wrong in two ways:

  * It used `DEFAULT_CONFIG` — the dataclass defaults baked into
    hsv_skin_filter.py — while run_phase7_4_batch.py loads the FROZEN config
    from configs/hsv_skin_thresholds.json. Those differ: the batch run uses
    hsv_source="original" (skin decided on the pre-SSR crop), the dataclass
    default is "ssr". v2 therefore filtered on a different image than the
    pipeline it was supposed to match, and the guard written to catch exactly
    that mismatch never fired, because it was reading the same wrong config it
    was meant to check. Every tier-escalation figure v2 printed is void.

  * It recomputed masks at all. Phase 7.4 ALREADY PERSISTED THEM.
    run_phase7_4_batch.py packs both the geometric and the post-HSV skin mask
    into one PNG per image via regions.encode_label_map(), at
    data/processed/label_maps/<exact manifest filename>.png:

        bits 0-2 : region id, 0 = outside all regions, 1-5 per REGION_IDS
        bit  3   : 1 if the pixel survived Phase 7.4's HSV filter

v3 therefore stops recomputing and READS THE REAL MASKS, via the module's own
regions.decode_label_map(). Training now consumes the identical skin_mask that
the batch pipeline produced and that Phase 8.2 averaged CIELAB over — not an
approximation of it. The frozen config is still loaded and echoed at startup,
but only so the run is self-documenting; no thresholds are applied here.

This also removes a whole class of error: no JPEG-black tolerance to tune, no
tier logic to replicate, no chance of drifting from the batch run.

==========================================================================
WHAT WAS WRONG WITH THE ORIGINAL train.py (still fixed here)
==========================================================================
Audited against "HueView System Architecture and Research Instrument",
Stages 3-5 of the HueView Proposed Approach (Figures 10 and 11).

[FIX 1] NO HSV SKIN FILTERING AT ALL.
    The original loaded the regional patch JPGs and used
    `mask = patch.any(axis=2)` as its "skin mask". Those JPGs carry Phase
    7.4's GEOMETRIC mask only — filter_region() writes
    `masked_image[geometric_mask] = ssr_image[geometric_mask]` and keeps
    skin_mask as a separate attribute. The HSV refinement the manuscript
    requires in Stage 3 ("remove residual non-skin features such as
    eyebrows, eyelashes, hair, eyes, and lips") never reached training.
    That is the entire explanation for the geometric-trained weights.
    Now: the real skin mask, read from the label map.

[FIX 2] CIELAB VIA OpenCV's QUANTIZED ENCODING.
    Old: cv2.cvtColor(..., COLOR_RGB2LAB) on uint8, then rescaled. That is
    the 8-bit display encoding, not CIELAB. Phase 8.2's cielab_features.py
    uses skimage.color.rgb2lab and documents why; the manuscript (Stage 5)
    specifies "RGB -> XYZ -> CIELAB via a standard matrix conversion".
    Training and the batch pipeline were in different colour spaces.
    Magnitude is small on an identical mask (dL* 0.08, da* 0.33, db* 0.31)
    but it is a correctness and train/inference-consistency fix.
    Now: skimage, same as Phase 8.2.

[FIX 3] full_face USED A 3-D WHOLE-IMAGE LAB VECTOR.
    Manuscript Figure 11 calls for a "CIELAB Feature Vector (15 Regional
    Color Features)", and regions.py states REGION_ORDER is authoritative
    "because it is the order the 15-element CIELAB Feature Vector
    (3 values x 5 regions) is built in at Phase 8.2". The old code called
    cielab_mean(face, mask=None) — a 3-vector over the ENTIRE image,
    including hair, eyes and background — and built every config with
    lab_dim=3. Now: 15-D in REGION_ORDER, lab branch sized per config.

[FIX 4] ZERO-IMPUTATION BIASED TOWARD THE DARKEST CLASS.
    Old: cielab_mean returned np.zeros(3) for an empty region. (0,0,0) in
    CIELAB is pure black; cielab_features.py warns it "would bias the
    classifier toward the darkest SCC class". The `bad` counter was printed
    and ignored. Now: within-face imputation first, then the TRAIN-split
    mean. Validation is imputed from TRAIN statistics only.

[FIX 5] CHECKPOINT AND EARLY-STOPPING MONITORED DIFFERENT QUANTITIES.
    EarlyStopping watched val_loss with restore_best_weights while
    ModelCheckpoint watched val_accuracy, so the file on disk and the
    weights in memory could be two different models. (The old comment
    claimed ModelCheckpoint watched val_loss; it did not.) Both now
    monitor val_accuracy.

[FIX 6] CSVLogger used the default append=False with one filename for both
    fit() calls, so fine-tuning truncated the head-training log.

NOTE ON RETRAINING: fixes 1 and 3 change the model's input distribution.
Weights from before this revision are NOT comparable. inference/hueview.py
must be switched to the same label-map + skimage + 15-D path when these
weights are deployed.

KNOWN DEVIATION TO DOCUMENT: run_phase7_4_batch.py records sigma = 30 for
SSR normalisation; the manuscript specifies sigma = 80. That is upstream of
this file and cannot be fixed here — either the manuscript or the SSR run
needs to change, but they should not stay different.

--------------------------------------------------------------------------
ADAPTER SECTION - edit these to match your Phase 7/8 output layout, then
nothing below should need changing.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import pathlib
import warnings
from collections import Counter

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.color import rgb2lab
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------- ADAPTER ------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]          # -> ml_pipeline/

SPLIT_DIR = ROOT / "data/processed"            # train_hueview_usable.csv / val_hueview_usable.csv
FACE_DIR = SPLIT_DIR / "images_ssr"            # SSR faces (Phase 7.2)
LABEL_MAP_DIR = SPLIT_DIR / "label_maps"       # Phase 7.4 packed masks
COVERAGE_CSV = SPLIT_DIR / "phase7_4_full_coverage_stats.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "configs"
HSV_CONFIG_JSON = CONFIG_DIR / "hsv_skin_thresholds.json"

FNAME_COL = "filename"
LABEL_COL = "SCC_label"
IMG_SIZE = 224
N_CLASSES = 6

# Cohort folders the SSR images may be nested under, mirroring
# run_phase7_4_batch.resolve_ssr_path().
SSR_COHORTS = ["", "processed", "v5_processed", "c1_processed", "c2_processed"]

# Per Figure 10 -> Figure 11, the refined skin mask feeds the CNN as well as
# the CIELAB branch ("isolated, illumination-adjusted skin patches", Stage 4).
# Set False only to ablate the masking itself.
APPLY_SKIN_MASK_TO_CNN = True

# full_face is "the primary illumination-adjusted full-face image" — the
# whole-face control condition — so its CNN input is the unmasked SSR face.
MASK_FULL_FACE_IMAGE = False

# -------------------------- END ADAPTER ----------------------------------


# ---------------------------------------- the project's own shared contract
from .regions import (  # noqa: E402
    MIN_VALID_PIXELS,
    REGION_ORDER,
    decode_label_map,
)

REGION_ORDER = list(REGION_ORDER)
# full_face is a *configuration*, not a sixth anatomical region: its CIELAB
# vector is the concatenation of these five (5 x 3 = 15, per Figure 11).
REGIONS = REGION_ORDER + ["full_face"]

MISSING: Counter = Counter()


def load_hsv_config():
    """Load the FROZEN Phase 7.4 config, the same artifact the batch run used.

    Nothing here applies these thresholds — the masks were already computed and
    saved. It is loaded so the run is self-documenting and so a changed config
    is visible at the top of the log rather than discovered later.
    """
    try:
        from .hsv_skin_filter import FilterConfig
    except Exception:
        return None
    if not HSV_CONFIG_JSON.exists():
        warnings.warn(
            f"{HSV_CONFIG_JSON} not found. The label maps were produced from a "
            f"frozen config; without it this run cannot state which thresholds "
            f"produced the masks it is training on.",
            RuntimeWarning, stacklevel=2,
        )
        return None
    return FilterConfig.from_json(HSV_CONFIG_JSON)


def _fname_rel(fname: str) -> pathlib.Path:
    """Normalise a manifest filename to its path relative to images/.

    Mirrors run_phase7_4_batch.py: strip anything up to and including
    'images/', keep ' (2)' intact — in this dataset it marks the
    c2_processed copy, a genuinely different image.
    """
    s = str(fname).replace("\\", "/")
    if "images/" in s:
        s = s.split("images/")[-1]
    return pathlib.Path(s)


def load_face(fname: str):
    """Load the SSR-normalised 224x224 face (Phase 7.2)."""
    rel = _fname_rel(fname)
    for cohort in SSR_COHORTS:
        base = FACE_DIR / cohort if cohort else FACE_DIR
        for ext in (".jpg", ".png", ".jpeg", ".bmp"):
            p = base / rel.with_suffix(ext)
            if p.is_file():
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE),
                                     interpolation=cv2.INTER_AREA)
                return rgb.astype(np.uint8)
    return None


def load_label_map(fname: str):
    """Load Phase 7.4's packed mask PNG for this image."""
    p = LABEL_MAP_DIR / _fname_rel(fname).with_suffix(".png")
    if not p.is_file():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:                       # written as 3-channel by some builds
        m = m[..., 0]
    if m.shape[:2] != (IMG_SIZE, IMG_SIZE):
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.uint8)


def load_patches(fname: str):
    """Return Phase 7.4's five RegionPatches for one image, or None.

    This is the whole of [FIX 1]. decode_label_map() is the module's own
    inverse of the codec run_phase7_4_batch.py wrote with, so the skin_mask
    here is bit-identical to the one the batch pipeline produced — not a
    recomputation, not an approximation.
    """
    ssr = load_face(fname)
    if ssr is None:
        MISSING["ssr_image"] += 1
        return None
    label = load_label_map(fname)
    if label is None:
        MISSING["label_map"] += 1
        return None
    return decode_label_map(label, ssr), ssr


# Baseline's recipe. Phase 6.4 writes this; Phase 10 only reads it.
# "run_tag" is not a training hyperparameter: if set (e.g. "manuscript"),
# every output filename for this run gets a "_<run_tag>" suffix so an
# experiment never overwrites the control run's weights.
DEFAULT_CFG = {
    "seed": 42, "batch_size": 32,
    "lr_head": 1e-3, "lr_finetune": 1e-5,
    "epochs_head": 15, "epochs_finetune": 25,
    "es_patience": 5, "dropout": 0.3,
    "unfreeze_last_n": 30, "use_class_weight": True,
    "weight_decay": 1e-4, "run_tag": "",
}


def load_cfg() -> dict:
    p = CONFIG_DIR / "train_config.json"
    cfg = dict(DEFAULT_CFG)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    else:
        print(f"[warn] {p} not found — using defaults. These MUST match Phase 6.4.")
    return cfg


def tag_suffix(cfg: dict) -> str:
    tag = cfg.get("run_tag", "")
    return f"_{tag}" if tag else ""


# ---------------------------------------------------------------- CIELAB
def cielab_mean(rgb: np.ndarray, mask: np.ndarray):
    """[FIX 2] Mean [L*, a*, b*] over the skin pixels, in true CIELAB.

    skimage.color.rgb2lab (L* 0-100, a*/b* ~ -128..127) is the "RGB -> XYZ ->
    CIELAB standard matrix conversion" of manuscript Stage 5 and the function
    Phase 8.2 uses, so training and the batch pipeline now agree.

    Returns None when the region is below MIN_VALID_PIXELS; the caller imputes.
    Never returns zeros — (0,0,0) is pure black and would drag the classifier
    toward the darkest SCC class. [FIX 4]
    """
    if mask is None or int(mask.sum()) < MIN_VALID_PIXELS:
        return None
    px = rgb[mask].astype(np.float64) / 255.0              # (N, 3) in [0,1]
    lab = rgb2lab(px.reshape(-1, 1, 3)).reshape(-1, 3)     # each pixel a 1x1 image
    return lab.mean(axis=0).astype(np.float32)


def lab_vector(region: str, fname: str):
    """CIELAB feature vector for one image under one configuration.

    [FIX 3] single region -> 3; full_face -> 15, concatenated in REGION_ORDER.
    Returns (vector, valid_flags), one flag per component region.
    """
    dim = 3 if region != "full_face" else 15
    got = load_patches(fname)
    if got is None:
        n = 1 if region != "full_face" else 5
        return np.full(dim, np.nan, np.float32), np.zeros(n, bool)
    patches, _ = got

    if region != "full_face":
        p = patches[region]
        v = cielab_mean(p.image, p.skin_mask)
        if v is None:
            return np.full(3, np.nan, np.float32), np.array([False])
        return v, np.array([True])

    parts, flags = [], []
    for r in REGION_ORDER:
        p = patches[r]
        v = cielab_mean(p.image, p.skin_mask)
        flags.append(v is not None)
        parts.append(np.full(3, np.nan, np.float32) if v is None else v)

    # Within-face imputation first: a face's own regions are strongly
    # correlated in colour — the mildest assumption available, and the one
    # cielab_features.py calls IMPUTE_FACE_MEAN.
    parts = np.stack(parts)
    flags = np.array(flags)
    if flags.any() and not flags.all():
        parts[~flags] = parts[flags].mean(axis=0)
    return parts.reshape(-1).astype(np.float32), flags


def report_real_tiers(df: pd.DataFrame, region: str, tag: str):
    """Print Phase 7.4's OWN tier counts for these images, from its stats CSV.

    These are the authoritative numbers — produced by the batch run itself,
    not inferred here. They belong in the write-up; anything this training
    script might estimate on its own would be a second, weaker tally.
    """
    if not COVERAGE_CSV.exists():
        return
    try:
        cov = pd.read_csv(COVERAGE_CSV)
    except Exception:
        return
    if not {"filename", "region", "status"} <= set(cov.columns):
        return
    want = set(df[FNAME_COL])
    sub = cov[(cov["region"] == region) & (cov["filename"].isin(want))]
    if sub.empty:
        return
    counts = sub["status"].value_counts()
    total = int(counts.sum())
    parts = [f"{s}={n} ({100 * n / total:.1f}%)" for s, n in counts.items()]
    print(f"[7.4] {region}/{tag} status (from {COVERAGE_CSV.name}): " + "  ".join(parts))


def cache_lab(region: str, df: pd.DataFrame, tag: str):
    """Precompute the LAB vectors once.

    Geometric augmentation (flip, small rotation) leaves the mean over skin
    pixels essentially unchanged, so caching is safe and avoids recomputing
    every epoch.

    Keyed by region + split tag only, not by run_tag — the raw LAB values do
    not change when lr/dropout/unfreeze are tuned, so the cache is reusable
    across experiments.
    """
    out = RESULT_DIR / f"lab_cache_{region}_{tag}.npy"
    ok_out = RESULT_DIR / f"lab_cache_{region}_{tag}_usable.npy"
    if out.exists() and ok_out.exists():
        return np.load(out), np.load(ok_out)

    MISSING.clear()
    vecs, usable = [], []
    for fn in df[FNAME_COL]:
        v, flags = lab_vector(region, fn)
        vecs.append(v)
        usable.append(bool(flags.any()))
    arr = np.stack(vecs)
    ok = np.array(usable)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    np.save(ok_out, ok)

    print(f"[lab] {region}/{tag}: cached {len(arr)} vectors, dim={arr.shape[1]}, "
          f"{(~ok).sum()} unusable (imputed from train mean)")
    if MISSING:
        print(f"[lab] {region}/{tag} missing inputs: " +
              "  ".join(f"{k}={v}" for k, v in MISSING.items()))
    report_real_tiers(df, region, tag)
    return arr, ok


def impute_from_train(lab: np.ndarray, usable: np.ndarray, fallback: np.ndarray):
    """Fill rows with no usable region using TRAIN-split means. Validation uses
    the TRAINING fallback, so validation never sees its own statistics."""
    lab = lab.copy()
    if (~usable).any():
        lab[~usable] = fallback
    bad = ~np.isfinite(lab)                  # never hand NaN to Keras
    if bad.any():
        lab[bad] = np.take(fallback, np.where(bad)[1])
    return lab


# ------------------------------------------------------------- data feed
def augment(patch: np.ndarray, region: str, rng: np.random.Generator) -> np.ndarray:
    # Geometric only. Colour jitter would partially undo SSR and contradict the
    # illumination-invariance claim — document this asymmetry vs Baseline.
    if region not in ("left_cheek", "right_cheek") and rng.random() < 0.5:
        patch = patch[:, ::-1]           # a flipped left cheek IS a right cheek
    ang = float(rng.uniform(-15, 15))
    M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), ang, 1.0)
    return cv2.warpAffine(patch, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class HueViewSeq(keras.utils.Sequence):
    def __init__(self, df, region, lab, scaler, batch, shuffle=False,
                 do_augment=False, seed=42, zero_lab=False):
        self.df = df.reset_index(drop=True)
        self.region, self.batch = region, batch
        self.lab = scaler.transform(lab).astype(np.float32)
        self.y = self.df[LABEL_COL].str.extract(r'(\d+)', expand=False).astype(int).to_numpy() - 1
        self.shuffle, self.do_augment, self.zero_lab = shuffle, do_augment, zero_lab
        self.rng = np.random.default_rng(seed)
        self.idx = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.idx)

    def __getitem__(self, i):
        sel = self.idx[i * self.batch:(i + 1) * self.batch]
        imgs = np.zeros((len(sel), IMG_SIZE, IMG_SIZE, 3), np.float32)
        for k, j in enumerate(sel):
            got = load_patches(self.df.at[j, FNAME_COL])
            if got is None:
                continue                      # left as zeros; counted in cache_lab
            patches, ssr = got
            if self.region == "full_face":
                img = ssr                     # whole-face control condition
                if MASK_FULL_FACE_IMAGE:
                    any_skin = np.zeros(ssr.shape[:2], bool)
                    for r in REGION_ORDER:
                        any_skin |= patches[r].skin_mask
                    img = ssr * any_skin[..., None]
            else:
                p = patches[self.region]
                # [FIX 1] Stage 4 receives "isolated, illumination-adjusted
                # skin patches": every non-skin pixel zeroed, using Phase
                # 7.4's own mask.
                img = p.image * p.skin_mask[..., None] if APPLY_SKIN_MASK_TO_CNN else p.image
            if self.do_augment:
                img = augment(img, self.region, self.rng)
            imgs[k] = img                     # 0..255 — EfficientNet wants this
        lab = self.lab[sel]
        if self.zero_lab:
            lab = np.zeros_like(lab)
        return {"img": imgs, "lab": lab}, keras.utils.to_categorical(self.y[sel], N_CLASSES)


# ----------------------------------------------------------------- model
def build_model(cfg, lab_dim=3):
    """If Phase 8.3 already defines this, import it instead and delete this.
    The head must mirror Baseline's 6.2 head so only the input differs.

    [FIX 3] lab_dim per configuration: 3 for a single region, 15 for full_face.
    """
    img_in = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="img")
    lab_in = keras.Input((lab_dim,), name="lab")

    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=img_in)
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(cfg["dropout"])(x)
    l = layers.Dense(32, activation="relu")(lab_in)
    x = layers.Concatenate()([x, l])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(cfg["dropout"])(x)
    out = layers.Dense(N_CLASSES, activation="softmax")(x)
    return keras.Model([img_in, lab_in], out), base


def compile_model(model, lr, weight_decay=0.0):
    if weight_decay > 0:
        optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])


# ------------------------------------------------------------------ run
def train_region(region, cfg, train_df, val_df, smoke=False, zero_lab=False):
    keras.utils.set_random_seed(cfg["seed"])      # reset per region

    if smoke:
        train_df, val_df = train_df.head(32), val_df.head(32)

    lab_tr, ok_tr = cache_lab(region, train_df, "train" if not smoke else "smoke")
    lab_va, ok_va = cache_lab(region, val_df, "val" if not smoke else "smokeval")

    if not ok_tr.any():
        raise RuntimeError(
            f"[{region}] no training image produced a usable skin mask. "
            f"Check that {LABEL_MAP_DIR} and {FACE_DIR} are populated and that "
            f"their filenames match {FNAME_COL} in the split CSVs."
        )
    fallback = np.nanmean(lab_tr[ok_tr], axis=0)
    lab_tr = impute_from_train(lab_tr, ok_tr, fallback)
    lab_va = impute_from_train(lab_va, ok_va, fallback)

    scaler = StandardScaler().fit(lab_tr)          # fit on TRAIN ONLY
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    suffix = tag_suffix(cfg)
    joblib.dump(scaler, MODEL_DIR / f"lab_scaler_{region}{suffix}.pkl")

    tr = HueViewSeq(train_df, region, lab_tr, scaler, cfg["batch_size"],
                    shuffle=True, do_augment=not smoke, seed=cfg["seed"],
                    zero_lab=zero_lab)
    va = HueViewSeq(val_df, region, lab_va, scaler, cfg["batch_size"],
                    zero_lab=zero_lab)

    cw = None
    if cfg["use_class_weight"] and not smoke:
        y = train_df[LABEL_COL].str.extract(r'(\d+)', expand=False).astype(int).to_numpy() - 1
        w = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y)
        cw = dict(enumerate(w))

    model, base = build_model(cfg, lab_dim=lab_tr.shape[1])
    ckpt = MODEL_DIR / f"hueview_{region}{suffix}.h5"

    # [FIX 5] Both callbacks monitor val_accuracy, so the weights restored in
    # memory, the file on disk and the logged best_val_accuracy are the same
    # model. [FIX 6] append=True so fine-tuning does not truncate the head log.
    def callbacks():
        return [
            keras.callbacks.EarlyStopping("val_accuracy", mode="max",
                                          patience=cfg["es_patience"],
                                          restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy",
                                            mode="max", save_best_only=True),
            keras.callbacks.CSVLogger(RESULT_DIR / f"trainlog_{region}{suffix}.csv",
                                      append=True),
        ]

    # stage 1 — frozen backbone
    compile_model(model, cfg["lr_head"], cfg["weight_decay"])
    h1 = model.fit(tr, validation_data=va, epochs=1 if smoke else cfg["epochs_head"],
                   class_weight=cw, callbacks=callbacks(), verbose=1)

    # stage 2 — fine-tune the top of EfficientNetB0
    base.trainable = True
    for layer in base.layers[:-cfg["unfreeze_last_n"]]:
        layer.trainable = False
    for layer in base.layers:                      # keep BN frozen
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, cfg["lr_finetune"], cfg["weight_decay"])
    h2 = model.fit(tr, validation_data=va, epochs=50 if smoke else cfg["epochs_finetune"],
                   class_weight=cw, callbacks=callbacks(), verbose=1)

    # No unconditional model.save(ckpt): ModelCheckpoint(save_best_only) already
    # holds the best-val_accuracy weights; saving again would overwrite them
    # with the last epoch's.
    hist = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in h2.history}
    (RESULT_DIR / f"history_{region}{suffix}.json").write_text(json.dumps(hist))
    best_acc = float(max(hist["val_accuracy"]))
    print(f"[done] {region}{suffix}: best val_acc={best_acc:.4f}  "
          f"best val_loss={float(np.min(hist['val_loss'])):.4f}  "
          f"lab_dim={lab_tr.shape[1]}  saved -> {ckpt.name}")
    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=REGIONS)
    ap.add_argument("--smoke", action="store_true",
                    help="overfit 32 images; should reach ~100%% train acc")
    ap.add_argument("--ablate-lab", action="store_true",
                    help="zero the LAB branch; accuracy should DROP")
    ap.add_argument("--cache-lab", action="store_true",
                    help="precompute LAB only (ignores --smoke; uses the full split)")
    a = ap.parse_args()

    cfg = load_cfg()
    if cfg.get("run_tag"):
        print(f"[cfg] run_tag={cfg['run_tag']!r} — outputs suffixed")

    hsv = load_hsv_config()
    if hsv is not None:
        print(f"[7.4] frozen config: {HSV_CONFIG_JSON.name}  "
              f"hsv_source={hsv.hsv_source}  "
              f"hue {hsv.primary.hue_min_deg:.0f}-{hsv.primary.hue_max_deg:.0f} deg  "
              f"sat {hsv.primary.sat_min}-{hsv.primary.sat_max}  "
              f"min_retention={hsv.min_retention}")
    print(f"[7.4] reading PRECOMPUTED masks from {LABEL_MAP_DIR.name}/ "
          f"(no thresholds applied here)")

    for d, what in ((LABEL_MAP_DIR, "label maps"), (FACE_DIR, "SSR images")):
        if not d.is_dir():
            raise SystemExit(
                f"ERROR: {what} not found at {d}\n"
                f"Run run_phase7_4_batch.py first — it writes the packed masks "
                f"this script trains on."
            )

    print(f"[cfg] regions={REGION_ORDER} (+ full_face, lab_dim 15)")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(SPLIT_DIR / "train_hueview_usable.csv")
    val_df = pd.read_csv(SPLIT_DIR / "val_hueview_usable.csv")
    print(f"train={len(train_df)}  val={len(val_df)}  tf={tf.__version__}")

    if a.cache_lab:
        for r in a.regions:
            cache_lab(r, train_df, "train")
            cache_lab(r, val_df, "val")
        return

    rows = []
    for r in a.regions:
        best = train_region(r, cfg, train_df, val_df,
                            smoke=a.smoke, zero_lab=a.ablate_lab)
        rows.append({"region": r, "best_val_accuracy": best})
    pd.DataFrame(rows).to_csv(RESULT_DIR / f"training_log{tag_suffix(cfg)}.csv",
                              index=False)


if __name__ == "__main__":
    main()