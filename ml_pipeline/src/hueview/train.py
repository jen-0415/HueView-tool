"""
PHASE 10 — HueView training.

Trains one fused (EfficientNetB0 + CIELAB) classifier per region configuration,
reusing the frozen Phase 5 split and the Baseline's training recipe.

Run order:
    python train_hueview.py --cache-lab          # once, per region
    python train_hueview.py --smoke              # 32-image overfit sanity check
    python train_hueview.py --ablate-lab         # confirms the LAB branch matters
    python train_hueview.py                      # the real run, all 6 configs

--------------------------------------------------------------------------
ADAPTER SECTION — edit these to match your Phase 7/8 output layout, then
nothing below should need changing.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------- ADAPTER ------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]

PATCH_DIR = ROOT / "data/processed/regions"    # JPG patches from Phase 7.4
SPLIT_DIR = ROOT / "data/processed"            # train_verified.csv / val_verified.csv
FACE_DIR = SPLIT_DIR / "images_ssr"            # SSR images (full faces)
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "configs"

REGIONS = ["forehead", "left_cheek", "right_cheek",
           "jawline", "nose_bridge", "full_face"]

FNAME_COL = "filename"
LABEL_COL = "SCC_label"
IMG_SIZE = 224
N_CLASSES = 6


def patch_paths(region: str, fname: str):
    """Phase 7.4 saved JPGs at: regions/MST-X/<stem>_<region>.jpg"""
    parts = pathlib.Path(fname)
    folder = parts.parent.name
    stem = parts.stem

    candidates = [
        PATCH_DIR / folder / f"{stem}_{region}.jpg",
    ]

    if stem.endswith(" (2)"):
        stem_no_dup = stem[:-4]
        candidates.append(
            PATCH_DIR / folder / f"{stem_no_dup}_{region}.jpg"
        )

    for path in candidates:
        if path.exists():
            return path, None

    return candidates[0], None


def load_patch(region: str, fname: str):
    """Load regional JPG, return (uint8 RGB 224x224, bool mask)."""
    if region == "full_face":
        # full_face uses the SSR full face images
        return load_face(fname), None

    p_img, _ = patch_paths(region, fname)
    bgr = cv2.imread(str(p_img))
    if bgr is None:
        raise FileNotFoundError(f"patch not found: {p_img}")
    patch = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if patch.shape[:2] != (IMG_SIZE, IMG_SIZE):
        patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # Mask: any non-black pixel (regions cropped to bounds by Phase 7.4)
    mask = patch.any(axis=2).astype(bool)
    return patch.astype(np.uint8), mask


def load_face(fname: str) -> np.ndarray:
    """Load SSR full face from images_ssr/MST-X/<stem>.jpg"""
    p0 = pathlib.Path(fname)
    folder = p0.parent.name
    stem = p0.stem
    for ext in (".jpg", ".png"):
        p = FACE_DIR / folder / f"{stem}{ext}"
        if p.exists():
            bgr = cv2.imread(str(p))
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            return img.astype(np.uint8)
    raise FileNotFoundError(f"face not found: {fname} in {FACE_DIR}")

# -------------------------- END ADAPTER ----------------------------------


# Baseline's recipe. Phase 6.4 writes this; Phase 10 only reads it.
# NOTE: added optional "run_tag" — not a training hyperparameter. If set in
# train_config.json (e.g. "expA"), all output filenames for this run get a
# "_<run_tag>" suffix so experiments never overwrite the control run's files.
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
    """'_<run_tag>' if a run_tag is set, else '' (keeps control-run filenames
    unchanged so nothing here breaks your existing outputs)."""
    tag = cfg.get("run_tag", "")
    return f"_{tag}" if tag else ""


# ---------------------------------------------------------------- CIELAB
def cielab_mean(patch_rgb: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Mean L*a*b* over valid skin pixels only, in real CIELAB units."""
    lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    if mask is None:
        px = lab.reshape(-1, 3)  # whole image
    else:
        px = lab[mask]
    if px.size == 0:
        return np.zeros(3, dtype=np.float32)
    return px.mean(axis=0).astype(np.float32)


def cache_lab(region: str, df: pd.DataFrame, tag: str) -> np.ndarray:
    """Precompute mean LAB per image once. Geometric augmentation (flip,
    small rotation) leaves the mean over skin pixels essentially unchanged,
    so caching is safe and saves recomputing it every epoch.

    NOTE: this cache is keyed by region + split tag only, not by run_tag —
    the raw LAB pixel values don't change when you tune lr/dropout/unfreeze,
    so it's safe (and much faster) to reuse across experiments A/B/C/E.
    It DOES need to be distinct from the --ablate-lab run in principle, but
    --ablate-lab zeroes the vector after loading the cache (see zero_lab in
    HueViewSeq), so the cached file itself never needs to differ."""
    out = RESULT_DIR / f"lab_cache_{region}_{tag}.npy"
    if out.exists():
        return np.load(out)
    vecs, bad = [], 0
    for fn in df[FNAME_COL]:
        patch, mask = load_patch(region, fn)
        if mask is not None and mask.sum() < 500:
            bad += 1
        vecs.append(cielab_mean(patch, mask))
    arr = np.stack(vecs)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print(f"[lab] {region}/{tag}: cached {len(arr)} vectors, {bad} with <500 skin px")
    return arr


# ------------------------------------------------------------- data feed
def augment(patch: np.ndarray, region: str, rng: np.random.Generator) -> np.ndarray:
    # Geometric only. Colour jitter would partially undo SSR and contradict
    # the illumination-invariance claim — document this asymmetry vs Baseline.
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
        imgs = np.empty((len(sel), IMG_SIZE, IMG_SIZE, 3), np.float32)
        for k, j in enumerate(sel):
            patch, _ = load_patch(self.region, self.df.at[j, FNAME_COL])
            if self.do_augment:
                patch = augment(patch, self.region, self.rng)
            imgs[k] = patch                       # 0..255 — EfficientNet wants this
        lab = self.lab[sel]
        if self.zero_lab:
            lab = np.zeros_like(lab)
        return {"img": imgs, "lab": lab}, keras.utils.to_categorical(self.y[sel], N_CLASSES)


# ----------------------------------------------------------------- model
def build_model(cfg, lab_dim=3):
    """If Phase 8.3 already defines this, import it instead and delete this.
    The head must mirror Baseline's 6.2 head so only the input differs."""
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
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = keras.optimizers.Adam(
            learning_rate=lr
        )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


# ------------------------------------------------------------------ run
def train_region(region, cfg, train_df, val_df, smoke=False, zero_lab=False):
    keras.utils.set_random_seed(cfg["seed"])      # reset per region

    if smoke:
        train_df, val_df = train_df.head(32), val_df.head(32)

    lab_tr = cache_lab(region, train_df, "train" if not smoke else "smoke")
    lab_va = cache_lab(region, val_df, "val" if not smoke else "smokeval")

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

    model, base = build_model(cfg)

    # --- FIX: filenames now carry the run_tag suffix so experiment runs
    # (expA, expB, ...) never collide with or overwrite the control run,
    # or each other. Default run_tag="" reproduces the original filenames.
    ckpt = MODEL_DIR / f"hueview_{region}{suffix}.h5"
    cbs = [
        keras.callbacks.EarlyStopping("val_loss", patience=cfg["es_patience"],
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", mode="max", save_best_only=True),
        keras.callbacks.CSVLogger(RESULT_DIR / f"trainlog_{region}{suffix}.csv"),
    ]

    # stage 1 — frozen backbone
    compile_model(model, cfg["lr_head"], cfg["weight_decay"])
    h1 = model.fit(tr, validation_data=va, epochs=1 if smoke else cfg["epochs_head"],
                   class_weight=cw, callbacks=cbs, verbose=1)

    # stage 2 — fine-tune the top of EfficientNetB0
    base.trainable = True
    for layer in base.layers[:-cfg["unfreeze_last_n"]]:
        layer.trainable = False
    for layer in base.layers:                      # keep BN frozen
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, cfg["lr_finetune"], cfg["weight_decay"])
    h2 = model.fit(tr, validation_data=va, epochs=50 if smoke else cfg["epochs_finetune"],
                   class_weight=cw, callbacks=cbs, verbose=1)

    # --- FIX: removed the unconditional `model.save(ckpt)` that used to sit
    # here. ModelCheckpoint(monitor="val_loss", save_best_only=True) already
    # wrote the best-val-loss weights to `ckpt` during training. Saving again
    # here would unconditionally overwrite that with the LAST epoch's weights
    # (which EarlyStopping's restore_best_weights may or may not match),
    # silently swapping in a worse model for evaluation. `ckpt` on disk is
    # now guaranteed to be the best-val-loss checkpoint from this run.
    hist = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in h2.history}
    (RESULT_DIR / f"history_{region}{suffix}.json").write_text(json.dumps(hist))
    best_loss = float(np.min(hist["val_loss"]))
    best_acc = float(max(hist["val_accuracy"]))
    print(
        f"[done] {region}{suffix}: "
        f"best val_acc={best_acc:.4f}  "
        f"best val_loss={best_loss:.4f}  "
        f"saved -> {ckpt.name}"
    )
    return best_acc 


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=REGIONS)
    ap.add_argument("--smoke", action="store_true",
                    help="overfit 32 images; should reach ~100%% train acc")
    ap.add_argument("--ablate-lab", action="store_true",
                    help="zero the LAB branch; accuracy should DROP")
    ap.add_argument("--cache-lab", action="store_true", help="precompute LAB only")
    a = ap.parse_args()

    cfg = load_cfg()
    if cfg.get("run_tag"):
        print(f"[cfg] run_tag={cfg['run_tag']!r} — outputs suffixed, control run untouched")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(SPLIT_DIR / "train_verified.csv")
    val_df = pd.read_csv(SPLIT_DIR / "val_verified.csv")
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
    suffix = tag_suffix(cfg)
    pd.DataFrame(rows).to_csv(RESULT_DIR / f"training_log{suffix}.csv", index=False)


if __name__ == "__main__":
    main()