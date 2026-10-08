"""
PHASE 11 — Evaluation.  [aligned with train.py MANUSCRIPT-ALIGNED REVISION v3]

Runs Baseline and all six HueView configurations over the frozen *usable* test
split in ONE pass, then writes:

    results/classification_results_record.csv   <- Table 10 (per-image, both models)
    results/metrics_overall.csv                 <- Table 15 (side-by-side)
    results/perclass_baseline.csv / perclass_hueview_<primary>.csv
    results/metrics_by_illumination.csv         <- SOP1 / SOP2
    results/metrics_by_region.csv               <- SOP3
    results/confusion_baseline.csv              <- Table 11
    results/confusion_hueview_<region>.csv      <- Table 12
    results/model_provenance.csv                <- which model files were evaluated
    results/metrics_summary.json

Phase 12 should read ONLY the per-image record and never re-run a model.

Run from HueView-tool/ (train.py uses relative imports, so -m is required):

    python -m ml_pipeline.src.hueview.evaluate
    python -m ml_pipeline.src.hueview.evaluate --primary full_face --run-tag final

How this matches train.py v3 — the HueView path reuses train.py's OWN code, so
test inputs are built exactly the way training inputs were:
  * Images: train.HueViewSeq (no shuffle, no augmentation) -> Phase 7.4 label-map
    skin masks, skin-masked regional patches, unmasked SSR face for full_face.
  * CIELAB: train.cache_lab -> skimage rgb2lab over the real skin mask;
    3-D per region, 15-D for full_face (REGION_ORDER), within-face imputation.
  * Images with no usable region are imputed with the TRAIN-split mean, via
    train.impute_from_train — the same fallback validation used.
  * Scaler: lab_scaler_<region>_<run_tag>.pkl, fit on TRAIN in train.py.
  * Models: hueview_<region>_<run_tag>.h5 (ModelCheckpoint best val_accuracy).

Baseline:
  * Reads the ORIGINAL (non-SSR) faces under data/processed/images/, because
    that is what Phase 6 trained it on. Do not point it at images_ssr/.

Checks before anything runs:
  * Every model/scaler exists; their paths, dates and hashes are recorded.
  * No test image is in HueView's or Baseline's train/val splits.
  * test_hueview_usable.csv is a subset of baseline_rgb_test.csv, so both
    models are scored on identical images (required for paired McNemar).

Full Face (manuscript, Fused Classification & Output): majority vote over the
five region-specific predictions; a tie goes to the class with the highest
mean softmax probability across the five regions. It is NOT a separately
trained model, so hueview_full_face_*.h5 is not loaded. Each region's softmax
is saved in the record (hv_<region>_p1..p6) so the vote can be re-derived.

Undertone: hue angle per facial region from the RAW (un-imputed) CIELAB mean;
majority vote across the FIVE facial regions only (full_face excluded), and a
region with no usable skin pixels does not vote.

NOTE: this overwrites results/ outputs. Back up the old folder first if you
still need the previous numbers.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tensorflow import keras

from .train import (FNAME_COL, IMG_SIZE, LABEL_COL, MODEL_DIR, N_CLASSES,
                    REGION_ORDER, REGIONS, RESULT_DIR, SPLIT_DIR, HueViewSeq,
                    _fname_rel, cache_lab, impute_from_train)

LABELS = list(range(1, N_CLASSES + 1))          # SCC-1 .. SCC-6
BATCH = 32

# Undertone majority vote: the five facial regions only (manuscript Stage 6)
VOTE_REGIONS = list(REGION_ORDER)

# ---------------------------- ADAPTER ------------------------------------
# HueView splits — the SAME files train.py trained on
TEST_CSV = SPLIT_DIR / "test_hueview_usable.csv"
TRAIN_CSV = SPLIT_DIR / "train_hueview_usable.csv"
VAL_CSV = SPLIT_DIR / "val_hueview_usable.csv"

# Baseline splits — what Phase 6 trained the Baseline on
BASELINE_TRAIN_CSV = SPLIT_DIR / "baseline_rgb_train.csv"
BASELINE_VAL_CSV = SPLIT_DIR / "baseline_rgb_val.csv"
BASELINE_TEST_CSV = SPLIT_DIR / "baseline_rgb_test.csv"

# Baseline model: original (non-SSR) faces, as used in Phase 6
BASELINE_H5 = MODEL_DIR / "baseline_effnet.h5"
BASELINE_INPUTS = ("image", "rgb_features")     # rename if your Phase 6.2 inputs differ
BASELINE_FACE_DIR = SPLIT_DIR / "images"        # images/MST-X/<file>
RGB_COLS = ("r_mean", "g_mean", "b_mean")       # in baseline_rgb_*.csv, scaled 0..1
RGB_TOL = 0.02                                  # max |image mean - CSV mean| per channel

# filename in the split CSV  ->  verified file on disk; filled by resolve_baseline_faces()
BASELINE_PATHS: dict[str, pathlib.Path] = {}
# filename -> (r, g, b) means recorded by Phase 6 in baseline_rgb_test.csv
BASELINE_RGB: dict[str, np.ndarray] = {}


def _read_face(p: pathlib.Path):
    bgr = cv2.imread(str(p))
    if bgr is None:
        return None
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def _candidates(fname: str):
    """Possible on-disk names. On disk a ' (2)' copy was renamed to '_1'
    (e.g. 'X (2).jpg' -> 'X_1.jpg'), but the split CSVs kept ' (2)'."""
    rel = _fname_rel(fname)
    stems = [rel.stem]
    if rel.stem.endswith(" (2)"):
        stems.append(rel.stem[:-len(" (2)")] + "_1")
    for stem in stems:
        for ext in (".jpg", ".png", ".jpeg", ".bmp"):
            p = BASELINE_FACE_DIR / rel.parent / f"{stem}{ext}"
            if p.is_file():
                yield p


def resolve_baseline_faces(test: pd.DataFrame):
    """Map every test filename to its original face BEFORE any model runs.

    Name matching alone is not trusted: each candidate's mean RGB must match
    the r/g/b means Phase 6 recorded for that exact filename, so a renamed
    '(2)' copy can never be swapped with its sibling image.
    """
    b = pd.read_csv(BASELINE_TEST_CSV)
    assert set(RGB_COLS) <= set(b.columns), f"{BASELINE_TEST_CSV.name} lacks {RGB_COLS}"
    BASELINE_RGB.update({f: np.array(v, np.float32)
                         for f, v in zip(b[FNAME_COL], b[list(RGB_COLS)].to_numpy())})

    # A recorded RGB triple shared by two different filenames is not a usable
    # fingerprint: Phase 6 wrote one image's means under both names (seen for
    # 'MST-7/0800_0_0_0_01.jpg' and its ' (2)' sibling).
    rgb_key = b[list(RGB_COLS)].round(7).astype(str).agg("|".join, axis=1)
    shared = set(b.loc[rgb_key.duplicated(keep=False), FNAME_COL])

    not_found, mismatched, exceptions, renamed, worst = [], [], [], 0, 0.0
    for f in test[FNAME_COL]:
        want = BASELINE_RGB[f]
        best, best_d = None, np.inf
        for p in _candidates(f):
            img = _read_face(p)
            if img is None:
                continue
            d = float(np.abs(img.reshape(-1, 3).mean(0) / 255.0 - want).max())
            if d < best_d:
                best, best_d = p, d
        if best is None:
            not_found.append(f)
        elif best_d > RGB_TOL:
            exact = best.stem == _fname_rel(f).stem
            if f in shared and exact:
                # The CSV record can't arbitrate; use the file with the exact
                # name — the same image HueView reads for this row.
                BASELINE_PATHS[f] = best
                exceptions.append({"filename": f, "used_file": str(best),
                                   "max_rgb_diff": round(best_d, 4),
                                   "reason": "Phase 6 RGB record duplicated across "
                                             "filenames; exact-name file used"})
            else:
                mismatched.append((f, best.name, best_d))
        else:
            BASELINE_PATHS[f] = best
            worst = max(worst, best_d)
            renamed += best.stem != _fname_rel(f).stem

    if exceptions:
        pd.DataFrame(exceptions).to_csv(RESULT_DIR / "baseline_rgb_exceptions.csv",
                                        index=False)
        for e in exceptions:
            print(f"[check] EXCEPTION (logged): {e['filename']} -> "
                  f"{pathlib.Path(e['used_file']).name} (max diff {e['max_rgb_diff']}); "
                  f"its RGB record is shared with another filename")

    if not_found or mismatched:
        for f in not_found[:5]:
            print(f"[check] NOT FOUND: {f}")
        for f, got, d in mismatched[:5]:
            print(f"[check] RGB MISMATCH: {f} -> {got} (max diff {d:.4f})")
        raise SystemExit(f"[check] Baseline faces: {len(not_found)} not found, "
                         f"{len(mismatched)} with RGB not matching "
                         f"{BASELINE_TEST_CSV.name}. Nothing was run.")
    print(f"[check] all {len(test)} Baseline faces found; "
          f"{len(test) - len(exceptions)} RGB-verified "
          f"({renamed} via ' (2)' -> '_1'; max diff {worst:.4f}), "
          f"{len(exceptions)} logged exception(s)")


def load_baseline_face(fname: str) -> np.ndarray:
    """Load the verified original (non-SSR) 224x224 face for the Baseline model."""
    img = _read_face(BASELINE_PATHS[fname])
    if img is None:
        raise FileNotFoundError(f"could not read {BASELINE_PATHS[fname]}")
    return img
# -------------------------- END ADAPTER ----------------------------------


def suffix_of(run_tag: str) -> str:
    return f"_{run_tag}" if run_tag else ""


def hv_model_path(region: str, run_tag: str):
    return MODEL_DIR / f"hueview_{region}{suffix_of(run_tag)}.h5"


def hv_scaler_path(region: str, run_tag: str):
    return MODEL_DIR / f"lab_scaler_{region}{suffix_of(run_tag)}.pkl"


# ------------------------------------------------------------- undertone
def baseline_undertone(rgb_mean: np.ndarray) -> str:
    """Phase 6.3 rule, on normalised RGB ratios (Aarabi et al., 2015)."""
    s = float(rgb_mean.sum()) or 1.0
    b = float(rgb_mean[2]) / s
    if b > 0.285:
        return "Cool"
    if b >= 0.275:
        return "Neutral"
    return "Warm"


def hue_angle(lab: np.ndarray) -> float:
    """h_ab in degrees, 0..360, from real-unit a*/b* (a 3-D [L*, a*, b*] vector)."""
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0)


def region_undertone(h: float) -> str:
    if h > 180.0:            # a*>0, b*<0 is pink, not "> 65 deg" (undertone.py decision c)
        h -= 360.0
    if h > 65.0:
        return "Warm"
    if h >= 55.0:
        return "Neutral"
    return "Cool"


def majority_undertone(per_region: dict[str, tuple[str, float]]) -> str:
    """Phase 9 majority vote; ties broken by the region closest to 60 deg.
    Regions without usable skin pixels are left out of per_region."""
    if not per_region:
        return "Undetermined"
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


# ------------------------------------------------------------- CIELAB
def clear_stale_test_cache():
    """Test LAB caches from an earlier SSR setting, mask version or test split
    would silently feed the wrong colour features. Always rebuild them.
    (cache_lab writes lab_cache_<region>_test.npy and ..._test_usable.npy.)"""
    for pat in ("lab_cache_*_test.npy", "lab_cache_*_test_usable.npy"):
        for f in RESULT_DIR.glob(pat):
            f.unlink()
            print(f"[lab] removed stale {f.name}")


def train_fallback(region: str, train_df: pd.DataFrame) -> np.ndarray:
    """Exactly train.train_region's fallback: TRAIN-split mean over usable rows.
    Uses the lab_cache_<region>_train.npy that training already wrote."""
    lab_tr, ok_tr = cache_lab(region, train_df, "train")
    assert len(lab_tr) == len(train_df), (
        f"lab_cache_{region}_train.npy has {len(lab_tr)} rows but "
        f"{TRAIN_CSV.name} has {len(train_df)} — stale train cache; delete it "
        f"and rerun (it was built from a different split).")
    return np.nanmean(lab_tr[ok_tr], axis=0)


# ------------------------------------------------------------- inference
def predict_baseline(df: pd.DataFrame):
    model = keras.models.load_model(BASELINE_H5, compile=False)
    preds, tones = [], []
    for i in range(0, len(df), BATCH):
        chunk = df[FNAME_COL].iloc[i:i + BATCH].tolist()
        imgs = np.stack([load_baseline_face(f) for f in chunk]).astype(np.float32)
        rgb = imgs.reshape(len(chunk), -1, 3).mean(axis=1) / 255.0
        p = model.predict({BASELINE_INPUTS[0]: imgs, BASELINE_INPUTS[1]: rgb}, verbose=0)
        preds.extend(p.argmax(1) + 1)
        tones.extend(baseline_undertone(r * 255.0) for r in rgb)
    return np.array(preds), tones


def predict_hueview(test_df, train_df, region, run_tag):
    """Returns (softmax, n x 6; raw test LAB; usable flags)."""
    model = keras.models.load_model(hv_model_path(region, run_tag), compile=False)
    scaler = joblib.load(hv_scaler_path(region, run_tag))   # fit on TRAIN in train.py

    lab_raw, ok = cache_lab(region, test_df, "test")         # same code as train/val
    lab = impute_from_train(lab_raw, ok, train_fallback(region, train_df))

    # Same input builder as training/validation: no shuffle, no augmentation.
    seq = HueViewSeq(test_df, region, lab, scaler, BATCH,
                     shuffle=False, do_augment=False)
    probs = []
    for i in range(len(seq)):
        x, _ = seq[i]
        probs.append(model.predict(x, verbose=0))
    return np.concatenate(probs), lab_raw, ok


def full_face_vote(probs: np.ndarray) -> np.ndarray:
    """Manuscript Full Face: majority vote of the five regions' predicted SCC;
    a tie goes to the tied class with the highest mean softmax across the five.
    probs: (n_images, 5 regions, 6 classes). Returns SCC labels 1..6."""
    n, _, k = probs.shape
    votes = np.zeros((n, k), dtype=int)
    np.add.at(votes, (np.arange(n)[:, None], probs.argmax(2)), 1)
    mean = probs.mean(1)
    mean[votes < votes.max(1, keepdims=True)] = -np.inf     # only tied leaders compete
    return mean.argmax(1) + 1


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
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),
        "f1": f1_score(y_true, y_pred, **kw),
    })


def save_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(cm, index=[f"true_SCC-{c}" for c in LABELS],
                 columns=[f"pred_SCC-{c}" for c in LABELS]).to_csv(path)


# ---------------------------------------------------------------- checks
def model_provenance(run_tag: str, baseline_row: dict | None = None):
    """Print and save exactly which model/scaler files were evaluated.
    baseline_row: the Baseline's row from an earlier provenance file, when its
    predictions are reused (--reuse-baseline) instead of re-run."""
    files = ([] if baseline_row else [BASELINE_H5]) + [hv_model_path(r, run_tag) for r in VOTE_REGIONS] \
          + [hv_scaler_path(r, run_tag) for r in VOTE_REGIONS]
    rows = [baseline_row] if baseline_row else []
    if baseline_row:
        print(f"[model] Baseline reused: {baseline_row['file']}  {baseline_row['sha256_12']}")
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        mt = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        rows.append({"file": str(p), "modified": mt.isoformat(timespec="seconds"),
                     "sha256_12": h})
        print(f"[model] {p.name:34s} {mt:%Y-%m-%d %H:%M}  {h}")
    pd.DataFrame(rows).to_csv(RESULT_DIR / "model_provenance.csv", index=False)

    # best val_accuracy logged by train.py for this run_tag, for cross-checking
    log = RESULT_DIR / f"training_log{suffix_of(run_tag)}.csv"
    if log.exists():
        print(f"[model] best val_accuracy from {log.name}:")
        print(pd.read_csv(log).to_string(index=False))


def check_splits(test: pd.DataFrame):
    """Leak check + paired-test check before any model runs."""
    test_set = set(test[FNAME_COL])

    # 1) no test image may appear in ANY split either model was trained/tuned on
    for path in (TRAIN_CSV, VAL_CSV, BASELINE_TRAIN_CSV, BASELINE_VAL_CSV):
        other = set(pd.read_csv(path)[FNAME_COL])
        overlap = other & test_set
        assert not overlap, f"LEAK: {len(overlap)} test files also in {path.name}"
    print("[check] no leakage into HueView or Baseline train/val splits")

    # 2) usable test must be a subset of the Baseline test split
    b_test = set(pd.read_csv(BASELINE_TEST_CSV)[FNAME_COL])
    extra = test_set - b_test
    assert not extra, (f"{len(extra)} usable test files are not in "
                       f"{BASELINE_TEST_CSV.name} — splits do not match")
    print(f"[check] usable test: {len(test_set)} / baseline test: {len(b_test)} "
          f"({len(b_test) - len(test_set)} excluded by pipeline-compatibility filter)")


def print_test_composition(rec: pd.DataFrame):
    """Counts to report in the manuscript (test set changed after filtering)."""
    print("\n[test] images per SCC:")
    print(rec["SCC_ground_truth"].value_counts().sort_index().to_string())
    print("\n[test] images per illumination bin:")
    print(rec["illumination_label"].value_counts().to_string())
    print()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="full_face", choices=REGIONS,
                    help="config used for the headline HueView column")
    ap.add_argument("--run-tag", default="final",
                    help="train.py run_tag of the models to evaluate "
                         "(hueview_<region>_<run_tag>.h5); '' for no suffix")
    ap.add_argument("--reuse-baseline", metavar="RECORD_CSV",
                    help="take baseline_pred / baseline_undertone from an earlier "
                         "classification_results_record.csv instead of running the "
                         "Baseline model (its model_provenance.csv must sit beside it)")
    a = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(TEST_CSV).reset_index(drop=True)
    train_df = pd.read_csv(TRAIN_CSV)
    n = len(test)
    print(f"test images: {n}  ({TEST_CSV.name})   run_tag={a.run_tag!r}")

    # --- every model must exist before we start a long run
    missing = [r for r in VOTE_REGIONS
               if not hv_model_path(r, a.run_tag).exists()
               or not hv_scaler_path(r, a.run_tag).exists()]
    assert not missing, (f"missing HueView model/scaler for {missing} "
                         f"(expected e.g. {hv_model_path(missing[0], a.run_tag).name} "
                         f"in {MODEL_DIR})" if missing else "")
    old_base, base_prov = None, None
    if a.reuse_baseline:
        src = pathlib.Path(a.reuse_baseline)
        old_base = pd.read_csv(src).set_index(FNAME_COL)
        prov = pd.read_csv(src.parent / "model_provenance.csv")
        base_prov = prov[prov["file"].str.contains("baseline")].iloc[0].to_dict()
        assert set(test[FNAME_COL]) == set(old_base.index), (
            f"{src.name} was scored on a different test set; cannot reuse its Baseline")
    else:
        assert BASELINE_H5.exists(), f"missing Baseline model: {BASELINE_H5}"
    model_provenance(a.run_tag, base_prov)

    check_splits(test)
    if not a.reuse_baseline:
        resolve_baseline_faces(test)
    clear_stale_test_cache()

    rec = pd.DataFrame({
        FNAME_COL: test[FNAME_COL],
        "SCC_ground_truth": test[LABEL_COL].astype(str)
                                           .str.extract(r'(\d+)', expand=False).astype(int),
        "illumination_label": test["illumination_label"],
    })
    print_test_composition(rec)

    if old_base is not None:
        b_pred = old_base.loc[test[FNAME_COL], "baseline_pred"].to_numpy()
        b_tone = old_base.loc[test[FNAME_COL], "baseline_undertone"].to_numpy()
    else:
        print("running baseline ...")
        b_pred, b_tone = predict_baseline(test)
    rec["baseline_pred"] = b_pred
    rec["baseline_undertone"] = b_tone

    raw_lab, usable, region_probs = {}, {}, []
    for r in VOTE_REGIONS:
        print(f"running hueview/{r} ...")
        probs, lab_raw, ok = predict_hueview(test, train_df, r, a.run_tag)
        rec[f"hv_{r}"] = probs.argmax(1) + 1
        region_probs.append(probs)
        raw_lab[r], usable[r] = lab_raw, ok
        print(f"[lab] {r}/test: {(~ok).sum()} of {n} images had no usable "
              f"skin region (imputed from train mean)")
    region_probs = np.stack(region_probs, axis=1)          # (n, 5, 6)
    rec["hv_full_face"] = full_face_vote(region_probs)
    top_votes = np.array([np.bincount(v, minlength=N_CLASSES).max()
                          for v in region_probs.argmax(2)])
    ties = sum(np.sum(np.bincount(v, minlength=N_CLASSES) == m) > 1
               for v, m in zip(region_probs.argmax(2), top_votes))
    print(f"[full_face] majority vote of {len(VOTE_REGIONS)} regions; "
          f"{ties} of {n} images needed the mean-softmax tie-break")

    # --- undertone: five facial regions, raw LAB, unusable regions abstain
    maj = []
    region_cols = {f"hv_{r}_{k}": [] for r in VOTE_REGIONS for k in ("hue", "undertone")}
    for i in range(n):
        votes = {}
        for r in VOTE_REGIONS:
            v = raw_lab[r][i]
            if usable[r][i] and np.isfinite(v).all():
                h = hue_angle(v)
                votes[r] = (region_undertone(h), h)
            h, u = votes.get(r, (None, np.nan))[::-1]
            region_cols[f"hv_{r}_hue"].append(h)
            region_cols[f"hv_{r}_undertone"].append(u or "")
        maj.append(majority_undertone(votes))
    rec["hv_majority_undertone"] = maj
    # Per-region descriptors (Appendix 5, Table 35); "" = region abstained
    for c, vals in region_cols.items():
        rec[c] = vals
    und = (rec["hv_majority_undertone"] == "Undetermined").sum()
    if und:
        print(f"[undertone] {und} images had no usable facial region -> 'Undetermined'")

    # --- integrity checks before anything downstream trusts this file
    assert len(rec) == n, "row count drifted"
    hue_cols = [c for c in rec.columns if c.endswith("_hue")]   # NaN = region abstained
    assert rec.drop(columns=hue_cols).notna().all().all(), "null predictions present"
    assert rec["SCC_ground_truth"].between(1, 6).all(), "bad ground truth"
    for j, r in enumerate(VOTE_REGIONS):
        for k in range(N_CLASSES):
            rec[f"hv_{r}_p{k + 1}"] = region_probs[:, j, k].round(6)
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

    with open(RESULT_DIR / "metrics_summary.json", "w") as fh:
        json.dump(overall.to_dict("records"), fh, indent=2)
    print("\nPhase 11 complete. Phase 12 reads classification_results_record.csv only.")


if __name__ == "__main__":
    main()