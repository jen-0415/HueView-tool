"""
PHASE 12 — Hypothesis testing (SOP4: H01, H02, H03).

Implements Chapter 3, Statistical Treatment, and Appendix 4 (Tables 29-33)
of the manuscript. Reads ONLY Phase 11's per-image record — no model is
loaded or re-run — so after retraining, re-run evaluate.py and then this
script; nothing here changes.

    results/classification_results_record.csv   <- input (evaluate.py)
    results/model_provenance.csv                <- input (evaluate.py), copied into the summary

    results/sop4/table29a_h01_paired_outcomes.csv   Baseline vs HueView, per illumination bin
    results/sop4/table29b_h02_paired_outcomes.csv   HueView region pairs
    results/sop4/table29c_h03_paired_outcomes.csv   Baseline vs HueView, full test set
    results/sop4/table30_bootstrap_summary.csv      bootstrap mean / SD / BCa CI per comparison
    results/sop4/table31a_h01.csv                   McNemar mid-p + BCa, Bonferroni 12
    results/sop4/table31b_h02_omnibus.csv           Cochran's Q + count of significant pairs
    results/sop4/table31c_h02_pairwise.csv          15 pairs x 4 metrics, Bonferroni 15
    results/sop4/table31d_h03.csv                   McNemar mid-p + BCa, Bonferroni 4
    results/sop4/table32a/b/c_*.csv                 performance improvement summary
    results/sop4/table33_decisions.csv              final decision per hypothesis x metric
    results/sop4/sop4_summary.md                    readable tables for Chapter 4
    results/sop4/table30_bootstrap_draws.csv.gz     every iteration (only with --save-draws)

Run from HueView-tool/:

    python -m ml_pipeline.src.hueview.significance
    python -m ml_pipeline.src.hueview.significance --resamples 10000 --save-draws

Procedures, as the manuscript states them:
  * Accuracy (H01, H03): McNemar's test with mid-p correction,
        mid-p = 2 * [binomcdf(n12, n, 0.5) - 0.5 * binompdf(n12, n, 0.5)],
    n12 = min(b, c), n = b + c (Fagerland et al., 2013).
  * Macro Precision / Recall / F1 (H01, H02, H03): paired bootstrap, B =
    10,000 resamples drawn with replacement STRATIFIED BY SCC, the same
    images for both configurations in each resample. Significant when the
    BCa interval of the paired difference excludes zero (Efron, 1987).
  * H02 accuracy: Cochran's Q across the six region configurations
    (chi-square, df = 5, alpha = 0.05), then the fifteen pairwise McNemar
    mid-p tests.
  * Bonferroni within each hypothesis family: H01 alpha = 0.05/12,
    H02 alpha = 0.05/15 per metric, H03 alpha = 0.05/4. BCa confidence
    levels 1 - alpha: 99.58 %, 99.67 %, 98.75 %. All tests two-sided.
  * Direction: Delta = HueView - Baseline (H01, H03) and Region A - Region B
    (H02). Positive favours HueView / Region A.
  * Macro averages use all six SCC labels with zero_division = 0, exactly
    evaluate.metric_row, so every point estimate equals the SOP1-3 tables.

Decision rule used for Table 33 (the manuscript does not spell it out —
confirm with the adviser): H01 and H02 are rejected for a metric when at
least one bin / pair is significant at its Bonferroni-adjusted level; H02
accuracy is rejected when Cochran's Q is significant.

HueView prediction used for H01/H03: hv_final, the manuscript's final SCC
(Fused Classification & Output): majority vote over the six configurations'
predictions, ties to the class with the highest mean softmax across the six.
H02 compares the six configurations themselves; its Full Face is hv_full_face,
the separately trained whole-face classifier. A record without hv_final
predates that rule and is refused.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, norm

ROOT = pathlib.Path(__file__).resolve().parents[2]          # -> ml_pipeline/
RESULT_DIR = ROOT / "results"
RECORD_CSV = RESULT_DIR / "classification_results_record.csv"
PROVENANCE_CSV = RESULT_DIR / "model_provenance.csv"
OUT_DIR = RESULT_DIR / "sop4"

N_CLASSES = 6
# Manuscript order (Appendix 3/4). The column names are evaluate.py's.
REGIONS = ["forehead", "left_cheek", "right_cheek", "jawline", "nose_bridge", "full_face"]
REGION_NAME = {"forehead": "Forehead", "left_cheek": "Left Cheek",
               "right_cheek": "Right Cheek", "jawline": "Jawline",
               "nose_bridge": "Nose Bridge", "full_face": "Full Face"}
BINS = ["Low", "Medium", "High"]
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
BOOT_METRICS = METRICS[1:]
PRETTY = {"accuracy": "Accuracy", "precision_macro": "Macro Precision",
          "recall_macro": "Macro Recall", "f1_macro": "Macro F1-Score"}
FAMILY_SIZE = {"H01": 12, "H02": 15, "H03": 4}
CHUNK = 500                                                  # bootstrap resamples per batch


# --------------------------------------------------------------------------
# Metrics from confusion matrices (vectorised over resamples)
# --------------------------------------------------------------------------
def metrics_from_cm(cm: np.ndarray) -> dict[str, np.ndarray]:
    """cm: (..., 6, 6), rows = truth. Macro over all six labels, zero_division=0."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    pred_n = cm.sum(axis=-2)
    true_n = cm.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_n > 0, tp / pred_n, 0.0)
        rec = np.where(true_n > 0, tp / true_n, 0.0)
        f1 = np.where(pred_n + true_n > 0, 2 * tp / (pred_n + true_n), 0.0)
    return {"accuracy": tp.sum(axis=-1) / cm.sum(axis=(-2, -1)),
            "precision_macro": prec.mean(axis=-1),
            "recall_macro": rec.mean(axis=-1),
            "f1_macro": f1.mean(axis=-1)}


def confusions(y, p, idx) -> np.ndarray:
    """Confusion matrices for each row of idx (B, n) -> (B, 6, 6)."""
    b = idx.shape[0]
    cell = (y[idx] - 1) * N_CLASSES + (p[idx] - 1)
    flat = cell + np.arange(b)[:, None] * N_CLASSES ** 2
    return np.bincount(flat.ravel(), minlength=b * N_CLASSES ** 2) \
             .reshape(b, N_CLASSES, N_CLASSES)


def point_metrics(y, p) -> dict[str, float]:
    m = metrics_from_cm(confusions(y, p, np.arange(len(y))[None, :]))
    return {k: float(v[0]) for k, v in m.items()}


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def paired_counts(y, a, b) -> dict[str, int]:
    ca, cb = a == y, b == y
    return {"both_correct": int((ca & cb).sum()), "a_only": int((ca & ~cb).sum()),
            "b_only": int((~ca & cb).sum()), "both_incorrect": int((~ca & ~cb).sum()),
            "total": int(len(y))}


def mcnemar_midp(b: int, c: int) -> float:
    """Manuscript formula: 2 * [cdf(n12) - 0.5 * pmf(n12)], n12 = min(b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    n12 = min(b, c)
    return float(min(1.0, 2 * (binom.cdf(n12, n, 0.5) - 0.5 * binom.pmf(n12, n, 0.5))))


def cochrans_q(correct: np.ndarray) -> tuple[float, float]:
    """correct: (n_images, k) 0/1. Returns (Q, p) against chi-square, df = k - 1."""
    k = correct.shape[1]
    cj = correct.sum(axis=0).astype(float)
    ri = correct.sum(axis=1).astype(float)
    n_tot = cj.sum()
    den = k * n_tot - (ri ** 2).sum()
    if den == 0:
        return float("nan"), 1.0
    q = (k - 1) * (k * (cj ** 2).sum() - n_tot ** 2) / den
    return float(q), float(chi2.sf(q, k - 1))


def stratified_boot_diffs(y, a, b, resamples, rng) -> dict[str, np.ndarray]:
    """Delta = metric(b) - metric(a) on resamples stratified by SCC (paired)."""
    strata = [np.flatnonzero(y == c) for c in np.unique(y)]
    out = {m: [] for m in BOOT_METRICS}
    done = 0
    while done < resamples:
        nb = min(CHUNK, resamples - done)
        idx = np.concatenate([s[rng.integers(0, len(s), size=(nb, len(s)))] for s in strata],
                             axis=1)
        ma = metrics_from_cm(confusions(y, a, idx))
        mb = metrics_from_cm(confusions(y, b, idx))
        for m in BOOT_METRICS:
            out[m].append(mb[m] - ma[m])
        done += nb
    return {m: np.concatenate(v) for m, v in out.items()}


def jackknife_accel(y, a, b, metric: str) -> float:
    """BCa acceleration from the leave-one-out jackknife. Images with the same
    (truth, pred_a, pred_b) give the same leave-one-out value, so only the
    distinct triples (at most 216) are evaluated."""
    cm_a = confusions(y, a, np.arange(len(y))[None, :])[0]
    cm_b = confusions(y, b, np.arange(len(y))[None, :])[0]
    triples, counts = np.unique(np.stack([y, a, b], axis=1), axis=0, return_counts=True)
    theta = np.empty(len(triples))
    for i, (t, pa, pb) in enumerate(triples):
        ca, cb = cm_a.copy(), cm_b.copy()
        ca[t - 1, pa - 1] -= 1
        cb[t - 1, pb - 1] -= 1
        theta[i] = float(metrics_from_cm(cb)[metric]) - float(metrics_from_cm(ca)[metric])
    d = np.average(theta, weights=counts) - theta
    den = 6 * (counts * d ** 2).sum() ** 1.5
    return float((counts * d ** 3).sum() / den) if den > 0 else 0.0


def bca_interval(draws, observed, accel, conf) -> tuple[float, float]:
    """Bias-corrected and accelerated percentile interval (Efron, 1987)."""
    b = len(draws)
    prop = (np.sum(draws < observed) + 0.5 * np.sum(draws == observed)) / b
    z0 = norm.ppf(np.clip(prop, 1 / (b + 1), b / (b + 1)))
    alpha = 1 - conf
    qs = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        qs.append(norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z))))
    lo, hi = np.percentile(draws, [100 * qs[0], 100 * qs[1]])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# One comparison: A vs B (Delta = B - A)
# --------------------------------------------------------------------------
def compare(y, a, b, family, resamples, rng, label_a, label_b, stratum, draws_out):
    alpha = 0.05 / FAMILY_SIZE[family]
    conf = 1 - alpha
    ma, mb = point_metrics(y, a), point_metrics(y, b)
    pc = paired_counts(y, a, b)
    rows = [{"family": family, "stratum": stratum, "a": label_a, "b": label_b,
             "metric": "accuracy", "test": "McNemar mid-p",
             "statistic": f"b = {pc['a_only']}, c = {pc['b_only']}",
             "score_a": ma["accuracy"], "score_b": mb["accuracy"],
             "observed_diff": mb["accuracy"] - ma["accuracy"],
             "bootstrap_mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
             "p_value": mcnemar_midp(pc["a_only"], pc["b_only"]),
             "alpha": alpha, "ci_level": np.nan}]
    rows[0]["reject_h0"] = rows[0]["p_value"] < alpha
    boot = []
    diffs = stratified_boot_diffs(y, a, b, resamples, rng)
    for m in BOOT_METRICS:
        obs = mb[m] - ma[m]
        lo, hi = bca_interval(diffs[m], obs, jackknife_accel(y, a, b, m), conf)
        rows.append({"family": family, "stratum": stratum, "a": label_a, "b": label_b,
                     "metric": m, "test": "Paired stratified bootstrap (BCa)",
                     "statistic": f"mean diff = {diffs[m].mean():+.4f}",
                     "score_a": ma[m], "score_b": mb[m], "observed_diff": obs,
                     "bootstrap_mean_diff": float(diffs[m].mean()),
                     "ci_low": lo, "ci_high": hi, "p_value": np.nan,
                     "alpha": alpha, "ci_level": conf,
                     "reject_h0": bool(lo > 0 or hi < 0)})
        boot.append({"family": family, "stratum": stratum, "a": label_a, "b": label_b,
                     "metric": m, "B": resamples, "observed_diff": obs,
                     "bootstrap_mean_diff": float(diffs[m].mean()),
                     "bootstrap_sd": float(diffs[m].std(ddof=1)),
                     "ci_level": conf, "bca_low": lo, "bca_high": hi})
        if draws_out is not None:
            draws_out.append(pd.DataFrame({
                "hypothesis_family": family, "stratum_or_pair": f"{stratum}: {label_a} vs {label_b}",
                "iteration": np.arange(1, resamples + 1), "metric": PRETTY[m],
                "difference_b_minus_a": diffs[m]}))
    for r in rows:
        d = r["observed_diff"]
        r["favours"] = (label_b if d > 0 else label_a) if r["reject_h0"] else ""
    return rows, boot, pc


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def provenance_lines() -> list[str]:
    if not PROVENANCE_CSV.exists():
        return ["model_provenance.csv not found — model files unknown."]
    prov = pd.read_csv(PROVENANCE_CSV)
    # PureWindowsPath splits on both / and \, since evaluate.py may have run on Windows
    return [f"{pathlib.PureWindowsPath(f).name}  sha256 {h}  ({m})"
            for f, h, m in zip(prov["file"], prov["sha256_12"], prov["modified"])]


def improvement(base: dict, other: dict) -> list[dict]:
    rows = []
    for m in METRICS:
        diff = other[m] - base[m]
        rows.append({"metric": PRETTY[m], "baseline": base[m], "hueview": other[m],
                     "absolute_difference": diff,
                     "relative_difference_pct": 100 * diff / base[m] if base[m] else np.nan})
    return rows


def md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) and not np.isnan(v)
                         else ("—" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def decision_text(r) -> str:
    return "Reject" if r["reject_h0"] else "Fail to Reject"


def stat_cell(r) -> str:
    if r["metric"] == "accuracy":
        return f"p = {r['p_value']:.4g}"
    return f"{100 * r['ci_level']:.2f}% CI: [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-draws", action="store_true",
                    help="write every bootstrap iteration (Table 30, large gzip CSV)")
    a = ap.parse_args()

    rec = pd.read_csv(RECORD_CSV)
    need = ["SCC_ground_truth", "illumination_label", "baseline_pred"] + [f"hv_{r}" for r in REGIONS]
    missing = [c for c in need if c not in rec.columns]
    assert not missing, f"{RECORD_CSV.name} is missing columns {missing} - re-run evaluate.py"
    assert rec[need].notna().all().all(), "null predictions present"
    for c in [need[0], need[2]] + need[3:]:
        assert rec[c].between(1, N_CLASSES).all(), f"{c} has labels outside 1..{N_CLASSES}"

    assert "hv_final" in rec.columns and rec["hv_final"].between(1, N_CLASSES).all(), (
        f"{RECORD_CSV.name} has no hv_final (the six-way vote) - re-run evaluate.py")
    hv_col = "hv_final"
    hv_note = ("HueView = final SCC, majority vote of the six configurations "
               "with mean-softmax tie-break (hv_final)")
    print(f"[sop4] {hv_note}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    draws = [] if a.save_draws else None
    y_all = rec["SCC_ground_truth"].to_numpy()
    base_all = rec["baseline_pred"].to_numpy()
    hv_all = rec[hv_col].to_numpy()

    # ---- H01: Baseline vs HueView within each illumination bin
    h01, boot, t29a = [], [], {}
    for bin_name in BINS:
        g = rec[rec["illumination_label"] == bin_name]
        if g.empty:
            print(f"[warn] no test images in illumination bin '{bin_name}' - skipped")
            continue
        rows, b, pc = compare(g["SCC_ground_truth"].to_numpy(), g["baseline_pred"].to_numpy(),
                              g[hv_col].to_numpy(), "H01", a.resamples, rng,
                              "Baseline", "HueView", bin_name, draws)
        h01 += rows
        boot += b
        t29a[f"{bin_name} Illumination"] = pc
        if len(g) < 200:
            print(f"[warn] '{bin_name}' bin has only {len(g)} test images - wide intervals expected")

    # ---- H03: Baseline vs HueView on the full test set
    h03, b, pc29c = compare(y_all, base_all, hv_all, "H03", a.resamples, rng,
                            "Baseline", "HueView", "Full test set", draws)
    boot += b

    # ---- H02: HueView across the six region configurations
    correct = np.stack([(rec[f"hv_{r}"].to_numpy() == y_all) for r in REGIONS], axis=1).astype(int)
    q, q_p = cochrans_q(correct)
    h02, t29b = [], []
    for ra, rb in combinations(REGIONS, 2):
        rows, b, pc = compare(y_all, rec[f"hv_{ra}"].to_numpy(), rec[f"hv_{rb}"].to_numpy(),
                              "H02", a.resamples, rng, REGION_NAME[ra], REGION_NAME[rb],
                              "All", draws)
        h02 += rows
        boot += b
        t29b.append({"pair": f"{REGION_NAME[ra]} vs. {REGION_NAME[rb]}",
                     "both_correct": pc["both_correct"], "a_correct_b_incorrect": pc["a_only"],
                     "a_incorrect_b_correct": pc["b_only"], "both_incorrect": pc["both_incorrect"],
                     "total": pc["total"]})
    q_sig = q_p < 0.05
    for r in h02:                                      # post-hoc only after a significant Q
        if r["metric"] == "accuracy":
            r["posthoc_run"] = q_sig
            if not q_sig:
                r["reject_h0"], r["favours"] = False, ""

    h01_df, h02_df, h03_df = pd.DataFrame(h01), pd.DataFrame(h02), pd.DataFrame(h03)

    # ---- Table 29
    t29a_df = pd.DataFrame(t29a)
    t29a_df.index = ["Both Baseline and HueView Correct", "Baseline Correct, HueView Incorrect",
                     "Baseline Incorrect, HueView Correct", "Both Baseline and HueView Incorrect",
                     "Total Test Images in Bin"]
    t29a_df.to_csv(OUT_DIR / "table29a_h01_paired_outcomes.csv", index_label="outcome")
    pd.DataFrame(t29b).to_csv(OUT_DIR / "table29b_h02_paired_outcomes.csv", index=False)
    pd.DataFrame({"outcome": t29a_df.index, "count": list(pc29c.values())}) \
      .to_csv(OUT_DIR / "table29c_h03_paired_outcomes.csv", index=False)

    # ---- Table 30
    pd.DataFrame(boot).to_csv(OUT_DIR / "table30_bootstrap_summary.csv", index=False)
    if draws is not None:
        pd.concat(draws).to_csv(OUT_DIR / "table30_bootstrap_draws.csv.gz",
                                index=False, compression="gzip")

    # ---- Table 31
    keep = ["stratum", "a", "b", "metric", "test", "statistic", "score_a", "score_b",
            "observed_diff", "bootstrap_mean_diff", "ci_low", "ci_high", "ci_level",
            "p_value", "alpha", "reject_h0", "favours"]
    h01_df[keep].to_csv(OUT_DIR / "table31a_h01.csv", index=False)
    h02_df[keep + ["posthoc_run"]].to_csv(OUT_DIR / "table31c_h02_pairwise.csv", index=False)
    h03_df[keep].to_csv(OUT_DIR / "table31d_h03.csv", index=False)
    omni = [{"metric": "Accuracy (omnibus)", "test": "Cochran's Q", "statistic": f"Q = {q:.3f}",
             "p_value": q_p, "alpha": 0.05, "reject_h0": q_sig}]
    for m in BOOT_METRICS:
        sub = h02_df[h02_df["metric"] == m]
        omni.append({"metric": f"{PRETTY[m]} (omnibus)", "test": "Bootstrap range across 15 pairs",
                     "statistic": f"significant pairs = {int(sub['reject_h0'].sum())}",
                     "p_value": np.nan, "alpha": 0.05 / 15, "reject_h0": bool(sub["reject_h0"].any())})
    omni_df = pd.DataFrame(omni)
    omni_df.to_csv(OUT_DIR / "table31b_h02_omnibus.csv", index=False)

    # ---- Table 32
    t32a = []
    for bin_name in BINS:
        g = rec[rec["illumination_label"] == bin_name]
        if g.empty:
            continue
        yy = g["SCC_ground_truth"].to_numpy()
        for row in improvement(point_metrics(yy, g["baseline_pred"].to_numpy()),
                               point_metrics(yy, g[hv_col].to_numpy())):
            t32a.append({"illumination": bin_name, **row})
    pd.DataFrame(t32a).to_csv(OUT_DIR / "table32a_improvement_by_illumination.csv", index=False)
    reg = pd.DataFrame([{"region": REGION_NAME[r],
                         **{PRETTY[m]: v for m, v in point_metrics(y_all, rec[f"hv_{r}"].to_numpy()).items()}}
                        for r in REGIONS])
    best = {"region": "Best Region", **{PRETTY[m]: reg.loc[reg[PRETTY[m]].idxmax(), "region"] for m in METRICS}}
    worst = {"region": "Worst Region", **{PRETTY[m]: reg.loc[reg[PRETTY[m]].idxmin(), "region"] for m in METRICS}}
    pd.concat([reg, pd.DataFrame([best, worst])]).to_csv(
        OUT_DIR / "table32b_regions.csv", index=False)
    t32c = pd.DataFrame(improvement(point_metrics(y_all, base_all), point_metrics(y_all, hv_all)))
    t32c.to_csv(OUT_DIR / "table32c_overall_improvement.csv", index=False)

    # ---- Table 33
    t33 = []
    for m in METRICS:
        s = h01_df[h01_df["metric"] == m]
        basis = ("McNemar's mid-p (per bin) vs. Bonferroni alpha = 0.0042" if m == "accuracy"
                 else "99.58% BCa CI (per bin) excludes / includes zero")
        t33.append({"hypothesis": f"H01 — {PRETTY[m]}", "basis": basis,
                    "bins_rejected": ", ".join(s.loc[s["reject_h0"], "stratum"]) or "none",
                    "conclusion": "Reject H0" if s["reject_h0"].any() else "Fail to Reject H0"})
    for m in METRICS:
        if m == "accuracy":
            basis = "Cochran's Q vs. alpha = 0.05; post-hoc McNemar's mid-p vs. alpha = 0.0033"
            rej = q_sig
        else:
            basis = "Pairwise 99.67% BCa CIs exclude / include zero"
            rej = bool(h02_df[(h02_df["metric"] == m)]["reject_h0"].any())
        t33.append({"hypothesis": f"H02 — {PRETTY[m]}", "basis": basis,
                    "bins_rejected": f"{int(h02_df[h02_df['metric'] == m]['reject_h0'].sum())} of 15 pairs",
                    "conclusion": "Reject H0" if rej else "Fail to Reject H0"})
    for _, r in h03_df.iterrows():
        basis = ("McNemar's mid-p vs. Bonferroni alpha = 0.0125" if r["metric"] == "accuracy"
                 else "98.75% BCa CI excludes / includes zero")
        t33.append({"hypothesis": f"H03 — {PRETTY[r['metric']]}", "basis": basis,
                    "bins_rejected": "full test set",
                    "conclusion": "Reject H0" if r["reject_h0"] else "Fail to Reject H0"})
    t33_df = pd.DataFrame(t33)
    t33_df.to_csv(OUT_DIR / "table33_decisions.csv", index=False)

    # ---- readable summary
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    md = ["# SOP4 — Hypothesis testing (Appendix 4)", "",
          f"- generated {stamp} from `{RECORD_CSV.name}` ({len(rec)} test images)",
          f"- B = {a.resamples} stratified bootstrap resamples, seed {a.seed}",
          f"- {hv_note}", "", "Models evaluated:", ""]
    md += [f"- `{l}`" for l in provenance_lines()] + [""]
    for title, df in (("Table 31.A — H01: Baseline vs. HueView, by illumination", h01_df),
                      ("Table 31.D — H03: Baseline vs. HueView, full test set", h03_df)):
        view = df.assign(Metric=df["metric"].map(PRETTY), Stratum=df["stratum"],
                         Baseline=df["score_a"], HueView=df["score_b"], Delta=df["observed_diff"],
                         Test=df["test"], Statistic=df["statistic"],
                         **{"p-value / BCa CI": df.apply(stat_cell, axis=1)},
                         Decision=df.apply(decision_text, axis=1), Favours=df["favours"])
        md += [f"## {title}", ""] + md_table(view, ["Stratum", "Metric", "Baseline", "HueView",
                                                   "Delta", "Statistic", "p-value / BCa CI",
                                                   "Decision", "Favours"])
    md += ["## Table 31.B — H02 omnibus", ""] + md_table(
        omni_df.assign(Decision=omni_df.apply(decision_text, axis=1)),
        ["metric", "test", "statistic", "p_value", "Decision"])
    view = h02_df.assign(Pair=h02_df["a"] + " vs. " + h02_df["b"], Metric=h02_df["metric"].map(PRETTY),
                         Delta=h02_df["observed_diff"],
                         **{"p-value / BCa CI": h02_df.apply(stat_cell, axis=1)},
                         Decision=h02_df.apply(decision_text, axis=1), Favours=h02_df["favours"])
    md += ["## Table 31.C — H02 pairwise (Delta = Region A − Region B)", ""] + md_table(
        view, ["Pair", "Metric", "Delta", "statistic", "p-value / BCa CI", "Decision", "Favours"])
    md += ["## Table 33 — Final decisions", ""] + md_table(
        t33_df, ["hypothesis", "basis", "bins_rejected", "conclusion"])
    md.append("Decision rule for H01/H02 (any bin / pair significant -> reject) is this "
              "script's reading of the manuscript; confirm with the adviser.")
    (OUT_DIR / "sop4_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(t33_df.to_string(index=False))
    print(f"\nCochran's Q = {q:.3f}, p = {q_p:.3g}")
    print(f"wrote {len(list(OUT_DIR.iterdir()))} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
