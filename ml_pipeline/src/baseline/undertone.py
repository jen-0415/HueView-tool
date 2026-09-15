"""
Phase 6.3 — Baseline Undertone Branch (rule-based, no training)
===============================================================

Classifies undertone from the same 3-value RGB vector Phase 6.1 produces.
No model, no training, no weights — a fixed threshold rule:

    r = R / (R+G+B)
    g = G / (R+G+B)
    b = B / (R+G+B)

    Cool     if b >  0.285
    Neutral  if 0.275 <= b <= 0.285
    Warm     if b <  0.275

Two things this branch is NOT:

  * It is not trained. Nothing here learns; the thresholds are fixed.
  * It is not scored. Per the methodology, undertone output is excluded
    from the Accuracy/Precision/Recall/F1 metrics that answer SOP1-SOP4.
    Those metrics cover SCC skin-tone classification only. Reporting an
    undertone accuracy alongside them would imply a validated ground
    truth this study doesn't have.

Scale-invariant by construction: the ratios normalize away overall
brightness, so it doesn't matter whether you pass [0.5, 0.4, 0.3] from
6.1 or raw [128, 102, 77]. Same answer either way.

Run directly to check how the thresholds split your data:
    python src/baseline/undertone.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

COOL_THRESHOLD = 0.285
WARM_THRESHOLD = 0.275

LABELS = ("Warm", "Neutral", "Cool")


def classify_undertone(rgb_vector) -> str:
    """
    Classify undertone from a 3-value RGB vector.

    Args:
        rgb_vector: [R, G, B] — either 6.1's [0,1] means or raw 0-255.
                    The ratios make scale irrelevant.

    Returns:
        "Warm", "Neutral", or "Cool".

    Raises:
        ValueError if the vector is the wrong length, negative, or sums
        to zero (a pure black patch has no meaningful undertone, and
        silently returning a label there would be worse than failing).
    """
    v = np.asarray(rgb_vector, dtype=np.float64).ravel()
    if v.size != 3:
        raise ValueError(f"Expected 3 values [R, G, B], got {v.size}")
    if np.any(v < 0):
        raise ValueError(f"Negative channel value in {v}")

    total = v.sum()
    if total <= 0:
        raise ValueError("R+G+B is zero — undertone is undefined for a black patch")

    b_ratio = v[2] / total

    if b_ratio > COOL_THRESHOLD:
        return "Cool"
    if b_ratio >= WARM_THRESHOLD:
        return "Neutral"
    return "Warm"


def undertone_ratios(rgb_vector):
    """Return (r, g, b) normalized ratios. Useful for inspection and plots."""
    v = np.asarray(rgb_vector, dtype=np.float64).ravel()
    total = v.sum()
    if total <= 0:
        raise ValueError("R+G+B is zero — ratios undefined")
    return tuple(v / total)


def classify_batch(features: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the rule across a DataFrame of 6.1 features.

    Expects columns: filename, r_mean, g_mean, b_mean
    Returns the same rows plus r_ratio, g_ratio, b_ratio, undertone.
    """
    required = {"r_mean", "g_mean", "b_mean"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    vals = features[["r_mean", "g_mean", "b_mean"]].to_numpy(dtype=np.float64)
    totals = vals.sum(axis=1)

    bad = totals <= 0
    if bad.any():
        print(f"[6.3] {bad.sum()} row(s) have R+G+B = 0; undertone left blank for those.")

    safe = np.where(totals > 0, totals, 1.0)
    ratios = vals / safe[:, None]

    b = ratios[:, 2]
    undertone = np.where(b > COOL_THRESHOLD, "Cool",
                         np.where(b >= WARM_THRESHOLD, "Neutral", "Warm"))
    undertone = np.where(bad, "", undertone)

    out = features.copy()
    out["r_ratio"] = ratios[:, 0]
    out["g_ratio"] = ratios[:, 1]
    out["b_ratio"] = ratios[:, 2]
    out["undertone"] = undertone
    return out


def report_distribution(classified: pd.DataFrame) -> None:
    """
    Show how the thresholds split the data.

    Worth looking at closely. The Neutral band is only 0.01 wide
    (0.275-0.285), so it's entirely possible almost everything lands in
    one bin. If it does, the rule isn't discriminating on your dataset —
    which is a finding to report, not a bug to hide. Check this before
    the thresholds appear in your results chapter.
    """
    n = len(classified)
    if n == 0:
        print("[6.3] Nothing to report.")
        return

    print("\nUndertone distribution:")
    counts = classified["undertone"].value_counts()
    for label in LABELS:
        c = int(counts.get(label, 0))
        print(f"    {label:<8} {c:>7}  ({100*c/n:5.1f}%)")
    blank = int(counts.get("", 0))
    if blank:
        print(f"    {'(none)':<8} {blank:>7}  ({100*blank/n:5.1f}%)")

    b = classified["b_ratio"].to_numpy()
    b = b[np.isfinite(b)]
    if b.size:
        print(f"\nb_ratio spread across {b.size} images:")
        print(f"    min {b.min():.4f}   max {b.max():.4f}")
        print(f"    mean {b.mean():.4f}   std {b.std():.4f}")
        for q in (5, 25, 50, 75, 95):
            print(f"    p{q:<3} {np.percentile(b, q):.4f}")
        print(f"\n    Thresholds sit at {WARM_THRESHOLD} and {COOL_THRESHOLD}.")

        dominant = counts.idxmax() if len(counts) else None
        if dominant and counts.max() / n > 0.9:
            print(f"\n    NOTE: {100*counts.max()/n:.0f}% of images fall in '{dominant}'.")
            print("    The rule is barely discriminating here. That's worth stating")
            print("    plainly in your results rather than presenting the three")
            print("    categories as though they were evenly exercised.")


if __name__ == "__main__":
    # Quick check on the rule itself, including both boundaries.
    print("Threshold behaviour (b_ratio -> label):\n")
    cases = [
        ([0.30, 0.35, 0.35], "strongly blue-dominant"),
        ([0.345, 0.345, 0.31], "clearly Cool"),
        ([0.355, 0.355, 0.29], "just inside Cool"),
        ([0.36, 0.36, 0.28], "mid Neutral band"),
        ([0.365, 0.365, 0.27], "just inside Warm"),
        ([0.40, 0.35, 0.25], "clearly Warm"),
        ([0.50, 0.30, 0.20], "strongly Warm"),
    ]
    for vec, note in cases:
        r, g, b = undertone_ratios(vec)
        print(f"    {str(vec):<24} b_ratio={b:.4f}  ->  {classify_undertone(vec):<8} ({note})")

    print("\nBoundary values (inclusive edges belong to Neutral):")
    for b_target in (0.2749, 0.2750, 0.2850, 0.2851):
        # construct a vector with exactly this b_ratio
        vec = [(1 - b_target) / 2, (1 - b_target) / 2, b_target]
        print(f"    b_ratio={b_target:.4f}  ->  {classify_undertone(vec)}")

    # If 6.1 output exists, summarize the real distribution.
    for name in ["baseline_rgb_train.csv", "baseline_rgb_test.csv"]:
        path = Path("data/processed") / name
        if path.exists():
            print(f"\n{'='*60}\nApplying to {name}\n{'='*60}")
            feats = pd.read_csv(path)
            classified = classify_batch(feats)
            report_distribution(classified)
            out = Path("data/processed") / name.replace("baseline_rgb_", "undertone_")
            classified.to_csv(out, index=False)
            print(f"\nWrote {out}")
            break
    else:
        print("\n(No 6.1 feature CSV found yet — run global_rgb_features.py first")
        print(" to see the distribution on your actual data.)")
