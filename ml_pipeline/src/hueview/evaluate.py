"""
PHASE 11 — Evaluation.

Runs Baseline and all six HueView configurations over the frozen test split in
ONE pass, then writes:

    results/classification_results_record.csv   <- Table 10 (per-image, both models)
    results/metrics_overall.csv                 <- Table 15 (side-by-side)
    results/metrics_by_illumination.csv         <- SOP1 / SOP2
    results/metrics_by_region.csv               <- SOP3
    results/confusion_baseline.csv              <- Table 11
    results/confusion_hueview_<region>.csv      <- Table 12
    results/metrics_<model>.json

Phase 12 should read ONLY the per-image record and never re-run a model.

    python evaluate_phase11.py
    python evaluate_phase11.py --primary full_face
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tensorflow import keras

from ml_pipeline.src.hueview.train import (CONFIG_DIR, FNAME_COL, IMG_SIZE, LABEL_COL, MODEL_DIR,
                    N_CLASSES, REGIONS, RESULT_DIR, SPLIT_DIR, cache_lab,
                    cielab_mean, load_patch)

LABELS = list(range(1, N_CLASSES + 1))          # SCC-1 .. SCC-6
BATCH = 32

# ---------------------------- ADAPTER ------------------------------------
FACE_DIR = SPLIT_DIR / "faces"       # Phase 3 output: 224x224 RGB cropped faces
FACE_SOURCE_DIRS = ["processed", "c1_processed", "c2_processed", "v5_processed"]
BASELINE_H5 = MODEL_DIR / "baseline_effnet.h5"
BASELINE_INPUTS = ("image", "rgb_features")     # rename if your Phase 6.2 inputs differ


def load_face(fname: str) -> np.ndarray:
    """Load full 224×224 face across four source folders."""
    stem = pathlib.Path(fname).stem
    folder = pathlib.Path(fname).parent.name
    for src in FACE_SOURCE_DIRS:
        for ext in (".jpg", ".png"):
            p = SPLIT_DIR / "images" / src / folder / f"{stem}{ext}"
            if p.exists():
                bgr = cv2.imread(str(p))
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                return img.astype(np.uint8)
    raise FileNotFoundError(f"face not found: {stem}")
# -------------------------- END ADAPTER ----------------------------------


# ------------------------------------------------------------- undertone
def baseline_undertone(rgb_mean: np.ndarray) -> str:
    """Phase 6.3 rule, on normalised RGB ratios."""
    s = float(rgb_mean.sum()) or 1.0
    b = float(rgb_mean[2]) / s
    if b > 0.285:
        return "Cool"
    if b >= 0.275:
        return "Neutral"
    return "Warm"


def hue_angle(lab: np.ndarray) -> float:
    """h_ab in degrees, 0..360, from real-unit a*/b*."""
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0)


def region_undertone(h: float) -> str:
    if h > 65.0:
        return "Warm"
    if h >= 55.0:
        return "Neutral"
    return "Cool"


def majority_undertone(per_region: dict[str, tuple[str, float]]) -> str:
    """Phase 9 majority vote; ties broken by the region closest to 60 deg."""
    votes = [v[0] for v in per_region.values()]
    counts = {u: votes.count(u) for u in set(votes)}
    top = max(counts.values())
    winners = [u for u, c in counts.items() if c == top]
    if len(winners) == 1:
        return winners[0]
    best, best_d = None, 1e9
    for u, h in per_region.values():
        if u in winners and abs(h - 60.0) < best_d:
            best, best_d = u, abs(h - 60.0)
    return best


# ------------------------------------------------------------- inference
def predict_baseline(df: pd.DataFrame):
    model = keras.models.load_model(BASELINE_H5)
    preds, tones = [], []
    for i in range(0, len(df), BATCH):
        chunk = df[FNAME_COL].iloc[i:i + BATCH].tolist()
        imgs = np.stack([load_face(f) for f in chunk]).astype(np.float32)
        rgb = imgs.reshape(len(chunk), -1, 3).mean(axis=1) / 255.0
        p = model.predict({BASELINE_INPUTS[0]: imgs, BASELINE_INPUTS[1]: rgb}, verbose=0)
        preds.extend(p.argmax(1) + 1)
        tones.extend(baseline_undertone(r * 255.0) for r in rgb)
    return np.array(preds), tones


def predict_hueview(df: pd.DataFrame, region: str):
    """Returns (SCC preds, per-image (undertone, hue_angle))."""
    model = keras.models.load_model(MODEL_DIR / f"hueview_{region}.h5")
    scaler = joblib.load(MODEL_DIR / f"lab_scaler_{region}.pkl")
    lab_raw = cache_lab(region, df, "test")
    lab = scaler.transform(lab_raw).astype(np.float32)

    preds = []
    for i in range(0, len(df), BATCH):
        chunk = df[FNAME_COL].iloc[i:i + BATCH].tolist()
        imgs = np.stack([load_patch(region, f)[0] for f in chunk]).astype(np.float32)
        p = model.predict({"img": imgs, "lab": lab[i:i + BATCH]}, verbose=0)
        preds.extend(p.argmax(1) + 1)

    tones = [(region_undertone(hue_angle(v)), hue_angle(v)) for v in lab_raw]
    return np.array(preds), tones


# --------------------------------------------------------------- metrics
def metric_row(name, y_true, y_pred):
    kw = dict(labels=LABELS, average="macro", zero_division=0)
    return {
        "model": name, "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, **kw),
        "recall_macro": recall_score(y_true, y_pred, **kw),
        "f1_macro": f1_score(y_true, y_pred, **kw),
    }


def per_class_table(y_true, y_pred):
    kw = dict(labels=LABELS, average=None, zero_division=0)
    return pd.DataFrame({
        "SCC": LABELS,
        "support": [int((y_true == c).sum()) for c in LABELS],
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),
        "f1": f1_score(y_true, y_pred, **kw),
    })


def save_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(cm, index=[f"true_SCC-{c}" for c in LABELS],
                 columns=[f"pred_SCC-{c}" for c in LABELS]).to_csv(path)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="full_face", choices=REGIONS,
                    help="config used for the headline HueView column")
    a = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(SPLIT_DIR / "test_verified.csv").reset_index(drop=True)
    n = len(test)
    print(f"test images: {n}")

    # --- leak re-check (cheap, do it here too, not only in Phase 5)
    for split in ("train", "val"):
        other = set(pd.read_csv(SPLIT_DIR / f"{split}.csv")[FNAME_COL])
        overlap = other & set(test[FNAME_COL])
        assert not overlap, f"LEAK: {len(overlap)} test files also in {split}.csv"

    rec = pd.DataFrame({
        FNAME_COL: test[FNAME_COL],
        "SCC_ground_truth": test[LABEL_COL].str.extract(r'(\d+)', expand=False).astype(int),
        "illumination_label": test["illumination_label"],
    })

    print("running baseline ...")
    b_pred, b_tone = predict_baseline(test)
    rec["baseline_pred"] = b_pred
    rec["baseline_undertone"] = b_tone

    tones_by_region = {}
    for r in REGIONS:
        print(f"running hueview/{r} ...")
        p, t = predict_hueview(test, r)
        rec[f"hv_{r}"] = p
        tones_by_region[r] = t

    rec["hv_majority_undertone"] = [
        majority_undertone({r: tones_by_region[r][i] for r in REGIONS})
        for i in range(n)
    ]

    # --- integrity checks before anything downstream trusts this file
    assert len(rec) == n, "row count drifted"
    assert rec.notna().all().all(), "null predictions present"
    assert rec["SCC_ground_truth"].between(1, 6).all(), "bad ground truth"
    rec.to_csv(RESULT_DIR / "classification_results_record.csv", index=False)
    print("wrote classification_results_record.csv")

    y = rec["SCC_ground_truth"].to_numpy()
    primary = f"hv_{a.primary}"

    # --- Table 15: overall side by side
    overall = pd.DataFrame([
        metric_row("Baseline", y, rec["baseline_pred"].to_numpy()),
        metric_row(f"HueView ({a.primary})", y, rec[primary].to_numpy()),
    ])
    overall.to_csv(RESULT_DIR / "metrics_overall.csv", index=False)
    print(overall.to_string(index=False))

    # --- per-class (report this, not just macro; SCC is imbalanced)
    per_class_table(y, rec["baseline_pred"].to_numpy()).to_csv(
        RESULT_DIR / "perclass_baseline.csv", index=False)
    per_class_table(y, rec[primary].to_numpy()).to_csv(
        RESULT_DIR / f"perclass_hueview_{a.primary}.csv", index=False)

    # --- SOP1 / SOP2: per-illumination
    rows = []
    for bin_name, g in rec.groupby("illumination_label"):
        yt = g["SCC_ground_truth"].to_numpy()
        for name, col in [("Baseline", "baseline_pred"),
                          (f"HueView ({a.primary})", primary)]:
            row = metric_row(name, yt, g[col].to_numpy())
            row["illumination"] = bin_name
            rows.append(row)
    illum = pd.DataFrame(rows)[["illumination", "model", "n", "accuracy",
                                "precision_macro", "recall_macro", "f1_macro"]]
    illum.to_csv(RESULT_DIR / "metrics_by_illumination.csv", index=False)
    print("\n", illum.to_string(index=False))
    for b, c in rec["illumination_label"].value_counts().items():
        if c < 200:
            print(f"[warn] illumination bin '{b}' has only {c} test images — "
                  f"metric differences there will be unstable in Phase 12.")

    # --- SOP3: per-region (HueView only)
    reg = pd.DataFrame([{**metric_row(r, y, rec[f"hv_{r}"].to_numpy()),
                         "region": r} for r in REGIONS])
    reg = reg[["region", "n", "accuracy", "precision_macro",
               "recall_macro", "f1_macro"]]
    reg.to_csv(RESULT_DIR / "metrics_by_region.csv", index=False)
    print("\n", reg.to_string(index=False))

    # --- confusion matrices, fixed label order for both
    save_confusion(y, rec["baseline_pred"].to_numpy(),
                   RESULT_DIR / "confusion_baseline.csv")
    for r in REGIONS:
        save_confusion(y, rec[f"hv_{r}"].to_numpy(),
                       RESULT_DIR / f"confusion_hueview_{r}.csv")

    json.dump(overall.to_dict("records"),
              open(RESULT_DIR / "metrics_summary.json", "w"), indent=2)
    print("\nPhase 11 complete. Phase 12 reads classification_results_record.csv only.")


if __name__ == "__main__":
    main()