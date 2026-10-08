"""
SUPPLEMENTARY ANALYSIS -- alternative fusion rules for HueView's six configurations.

NOT part of the pre-specified methodology. Chapter 3 fixes HueView's final SCC
as an equal-weight majority vote of the six configurations (hv_final in the
record); SOP 2 and SOP 4 report that rule and nothing here replaces it. This
script answers a Discussion / Recommendations question only: would a fusion
rule that does not let weaker configurations outvote stronger ones do better?

Two rules, both fixed here before any test result is looked at:
  * soft    -- average the six configurations' softmax vectors, take the argmax.
               No parameters, so nothing is tuned.
  * weighted -- each configuration's vote counts with weight = its macro F1 on
               the VALIDATION split; a tie in total weight goes to the tied class
               with the highest mean softmax (the same tie rule as hv_final).

Two stages, run in this order (the default runs both):

  1. --stage val
     Runs the six models over the validation split (val_hueview_usable.csv),
     the same input pipeline as evaluate.py, and writes
         results/supplementary/val_probs.npz
         results/supplementary/fusion_weights.json   <- locked: SHA-256 recorded
     The test split is not read in this stage.

  2. --stage test
     Reads the locked weights and the test softmax already saved by evaluate.py
     (classification_results_record.csv, hv_<config>_p1..p6). No model is run,
     so the test set is touched exactly once per rule. Writes
         results/supplementary/test_metrics.csv          overall + per illumination
         results/supplementary/test_comparisons.csv      McNemar mid-p + paired BCa
         results/supplementary/summary.md

     Each rule is compared with the Baseline and with the official equal-weight
     vote (hv_final). Bonferroni over every test in this file: 4 comparisons x 4
     metrics = 16, alpha = 0.05 / 16.

Run from HueView-tool/, after evaluate.py, on the machine the thesis numbers
come from (same images, models and environment):

    python -m ml_pipeline.src.hueview.supplementary_fusion --run-tag final1
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from . import significance as sig
from .evaluate import (BATCH, HEADS, TRAIN_CSV, VAL_CSV,
                       hv_model_path, hv_scaler_path, train_fallback)
from .train import (FNAME_COL, LABEL_COL, N_CLASSES, RESULT_DIR, HueViewSeq,
                    cache_lab, impute_from_train)

OUT_DIR = RESULT_DIR / "supplementary"
WEIGHTS_JSON = OUT_DIR / "fusion_weights.json"
VAL_NPZ = OUT_DIR / "val_probs.npz"
VAL_TAG = "val_fusion"            # own LAB cache name, rebuilt every run
LABELS = np.arange(1, N_CLASSES + 1)
FAMILY = "SUPP"
sig.FAMILY_SIZE[FAMILY] = 16      # 4 comparisons x 4 metrics


# ----------------------------------------------------------------- fusion
def soft_vote(probs: np.ndarray) -> np.ndarray:
    """probs (n, 6 configs, 6 classes) -> SCC 1..6: argmax of the mean softmax."""
    return probs.mean(1).argmax(1) + 1


def weighted_vote(probs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Each configuration's predicted class receives its weight; the class with
    the largest total wins, a tie going to the highest mean softmax."""
    n, k, c = probs.shape
    score = np.zeros((n, c))
    np.add.at(score, (np.arange(n)[:, None], probs.argmax(2)), np.broadcast_to(w, (n, k)))
    mean = probs.mean(1)
    lead = np.isclose(score, score.max(1, keepdims=True))
    mean[~lead] = -np.inf
    return mean.argmax(1) + 1


def equal_vote(probs: np.ndarray) -> np.ndarray:
    return weighted_vote(probs, np.ones(probs.shape[1]))


def sha12(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def scc_of(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL_COL].astype(str).str.extract(r"(\d+)", expand=False).astype(int).to_numpy()


# ------------------------------------------------------------ stage 1: val
def stage_val(run_tag: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(VAL_CSV).reset_index(drop=True)
    train_df = pd.read_csv(TRAIN_CSV)
    y = scc_of(val)
    print(f"[val] {len(val)} validation images ({VAL_CSV.name}), run_tag={run_tag!r}")

    for r in HEADS:   # never reuse a cache built under other settings
        for f in (RESULT_DIR / f"lab_cache_{r}_{VAL_TAG}.npy",
                  RESULT_DIR / f"lab_cache_{r}_{VAL_TAG}_usable.npy"):
            f.unlink(missing_ok=True)

    probs, files = [], {}
    for r in HEADS:
        print(f"[val] running {r} ...")
        model = keras.models.load_model(hv_model_path(r, run_tag), compile=False)
        scaler = joblib.load(hv_scaler_path(r, run_tag))
        lab_raw, ok = cache_lab(r, val, VAL_TAG)
        lab = impute_from_train(lab_raw, ok, train_fallback(r, train_df))
        seq = HueViewSeq(val, r, lab, scaler, BATCH, shuffle=False, do_augment=False)
        probs.append(np.concatenate([model.predict(seq[i][0], verbose=0) for i in range(len(seq))]))
        files[r] = {"model": hv_model_path(r, run_tag).name, "model_sha256_12": sha12(hv_model_path(r, run_tag)),
                    "scaler": hv_scaler_path(r, run_tag).name, "scaler_sha256_12": sha12(hv_scaler_path(r, run_tag))}
    probs = np.stack(probs, 1)                                   # (n, 6, 6)
    np.savez_compressed(VAL_NPZ, probs=probs, y=y, filenames=val[FNAME_COL].to_numpy())

    per_cfg = {r: sig.point_metrics(y, probs[:, j].argmax(1) + 1) for j, r in enumerate(HEADS)}
    weights = {r: per_cfg[r]["f1_macro"] for r in HEADS}
    w = np.array([weights[r] for r in HEADS])
    rules = {"equal-weight vote (official)": equal_vote(probs),
             "soft vote": soft_vote(probs), "weighted vote": weighted_vote(probs, w)}
    rows = [{"configuration_or_rule": r, **per_cfg[r]} for r in HEADS] + \
           [{"configuration_or_rule": k, **sig.point_metrics(y, p)} for k, p in rules.items()]
    pd.DataFrame(rows).to_csv(OUT_DIR / "validation_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    WEIGHTS_JSON.write_text(json.dumps({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "basis": "macro F1 of each configuration on the validation split",
        "run_tag": run_tag, "n_validation": int(len(val)),
        "order": HEADS, "weights": weights, "models": files,
    }, indent=2), encoding="utf-8")
    print(f"[val] weights locked in {WEIGHTS_JSON.name} (sha256 {sha12(WEIGHTS_JSON)})")


# ----------------------------------------------------------- stage 2: test
def stage_test(resamples: int, seed: int):
    assert WEIGHTS_JSON.exists(), "run --stage val first; the weights must exist before the test is read"
    spec = json.loads(WEIGHTS_JSON.read_text(encoding="utf-8"))
    assert spec["order"] == HEADS, f"weights were fixed for {spec['order']}, expected {HEADS}"
    w = np.array([spec["weights"][r] for r in HEADS])

    record = RESULT_DIR / "classification_results_record.csv"
    rec = pd.read_csv(record)
    need = [f"hv_{r}_p{k}" for r in HEADS for k in LABELS] + ["hv_final", "baseline_pred"]
    missing = [c for c in need if c not in rec.columns]
    assert not missing, f"{record.name} lacks {missing[:3]}... -- re-run evaluate.py"

    # The weights must come from the same models evaluate.py scored.
    prov = pd.read_csv(RESULT_DIR / "model_provenance.csv")
    scored = {pathlib_name(f): h for f, h in zip(prov["file"], prov["sha256_12"])}
    for r, f in spec["models"].items():
        assert scored.get(f["model"]) == f["model_sha256_12"], (
            f"{f['model']}: weights were fitted on a different file than evaluate.py scored")

    y = rec["SCC_ground_truth"].to_numpy()
    probs = np.stack([rec[[f"hv_{r}_p{k}" for k in LABELS]].to_numpy() for r in HEADS], 1)
    base, official = rec["baseline_pred"].to_numpy(), rec["hv_final"].to_numpy()
    assert (equal_vote(probs) == official).all(), "hv_final no longer matches the equal-weight rule"
    rules = {"Soft vote": soft_vote(probs), "Weighted vote": weighted_vote(probs, w)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = []
    for stratum, m in [("All", np.ones(len(y), bool))] + \
                      [(b, (rec["illumination_label"] == b).to_numpy()) for b in sig.BINS]:
        for name, p in [("Baseline", base), ("HueView equal-weight vote (official)", official),
                        *rules.items()]:
            metrics.append({"stratum": stratum, "model": name, "n": int(m.sum()),
                            **sig.point_metrics(y[m], p[m])})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT_DIR / "test_metrics.csv", index=False)

    rng = np.random.default_rng(seed)
    rows = []
    for name, p in rules.items():
        for ref_name, ref in [("Baseline", base), ("HueView equal-weight vote (official)", official)]:
            r_, _, _ = sig.compare(y, ref, p, FAMILY, resamples, rng, ref_name, name, "All", None)
            rows += r_
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_DIR / "test_comparisons.csv", index=False)

    write_summary(spec, metrics, comp, resamples)
    print(metrics[metrics.stratum == "All"].to_string(index=False))
    print(comp[["a", "b", "metric", "observed_diff", "p_value", "ci_low", "ci_high",
                "reject_h0", "favours"]].to_string(index=False))
    print(f"\nwrote {OUT_DIR}")


def pathlib_name(p: str) -> str:
    return p.replace("\\", "/").rsplit("/", 1)[-1]


def write_summary(spec, metrics, comp, resamples):
    pct = lambda v: f"{v * 100:.1f}%"
    alpha = 0.05 / sig.FAMILY_SIZE[FAMILY]
    lines = [
        "# Supplementary analysis: alternative fusion rules",
        "",
        "**Not part of the pre-specified methodology.** HueView's official result (SOP 2, SOP 4) is",
        "the equal-weight majority vote of the six configurations defined in Chapter 3. The rules",
        "below were fixed before the test set was read, with weights taken from the validation split.",
        "",
        f"- Weights fixed {spec['created']} from {spec['n_validation']} validation images "
        f"(file sha256 {sha12(WEIGHTS_JSON)}):",
        *[f"  - {r}: {w:.4f}" for r, w in spec["weights"].items()],
        f"- Tests: McNemar mid-p (accuracy) and paired stratified bootstrap with BCa intervals "
        f"(B = {resamples:,}) for macro precision, recall and F1; Bonferroni alpha = 0.05/16 = {alpha:.4f}.",
        "",
        "## Test-set performance (full test set)",
        "",
        "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
        "|---|---|---|---|---|",
    ]
    for _, r in metrics[metrics.stratum == "All"].iterrows():
        lines.append(f"| {r.model} | {pct(r.accuracy)} | {pct(r.precision_macro)} | "
                     f"{pct(r.recall_macro)} | {pct(r.f1_macro)} |")
    lines += ["", "## Paired comparisons", "",
              "| Comparison | Metric | Difference (B − A) | p / CI | Significant |", "|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        test = (f"p = {r.p_value:.2e}" if r.metric == "accuracy"
                else f"[{r.ci_low * 100:+.2f}, {r.ci_high * 100:+.2f}] pts")
        lines.append(f"| {r.b} vs {r.a} | {sig.PRETTY[r.metric]} | {r.observed_diff * 100:+.2f} pts | "
                     f"{test} | {'yes, favours ' + r.favours if r.reject_h0 else 'no'} |")
    lines += ["", "Report these as exploratory. The test set had already been evaluated with the",
              "official rule when this analysis was designed, so the comparison carries some hindsight;",
              "state this in the Limitations."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["val", "test", "both"], default="both")
    ap.add_argument("--run-tag", default="final1")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    if a.stage in ("val", "both"):
        stage_val(a.run_tag)
    if a.stage in ("test", "both"):
        stage_test(a.resamples, a.seed)


if __name__ == "__main__":
    main()
