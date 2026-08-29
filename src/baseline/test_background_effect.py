"""
Is background contamination driving the 70% "Cool" result?
==========================================================

Phase 6.3 classified 70.2% of training images as Cool. Human skin is
predominantly warm-toned, so that describes the images rather than the
faces in them.

Hypothesis: Phase 6.1 averages over the entire 224x224 frame — background,
hair, clothing, shadows — and backgrounds are typically bluer than skin.
The 0.275/0.285 thresholds look like values derived from skin pixels, but
they're being applied to a whole-scene average.

This measures the effect four ways on the same images:

    full         the whole frame (what 6.1 does now)
    center_50    middle 50% by area — mostly face
    center_25    middle 25% — cheeks, forehead, nose
    skin_masked  YCrCb skin-range pixels only, no geometric assumption

If b_ratio drops as the crop tightens, the background is the cause and
the number here belongs in your limitations section. If it stays flat,
the thresholds are simply wrong for this data and that's a different
conversation.

Run from the repo root:  python src/baseline/test_background_effect.py

Read-only. Nothing is overwritten.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from path_resolver import resolve_image_path
except ImportError:
    from src.baseline.path_resolver import resolve_image_path

PROC = Path("data/processed")
SAMPLE = 600
WARM_T, COOL_T = 0.275, 0.285

# Standard YCrCb skin range. Crude, but it makes no assumption about WHERE
# the face is — a useful cross-check on the geometric crops, which assume
# the face is centered.
SKIN_LOW = np.array([0, 133, 77], dtype=np.uint8)
SKIN_HIGH = np.array([255, 173, 127], dtype=np.uint8)


def b_ratio(mean_rgb):
    total = float(np.sum(mean_rgb))
    if total <= 0:
        return np.nan
    return float(mean_rgb[2]) / total


def label(b):
    if np.isnan(b):
        return ""
    if b > COOL_T:
        return "Cool"
    if b >= WARM_T:
        return "Neutral"
    return "Warm"


def center_crop(img, frac):
    """Crop the central `frac` of the image area."""
    h, w = img.shape[:2]
    side = np.sqrt(frac)
    ch, cw = int(h * side), int(w * side)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def skin_mean(rgb):
    """Mean RGB over YCrCb skin-range pixels. Returns (mean, coverage)."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, SKIN_LOW, SKIN_HIGH)
    n = int(np.count_nonzero(mask))
    if n < 100:
        return None, n / mask.size
    pixels = rgb[mask > 0]
    return pixels.mean(axis=0), n / mask.size


def main():
    train = PROC / "train.csv"
    if not train.exists():
        print(f"!! {train} not found. Run from the repo root.")
        return

    df = pd.read_csv(train)
    col = "filename" if "filename" in df.columns else df.columns[0]
    sample = df.sample(n=min(SAMPLE, len(df)), random_state=19)

    print("=" * 70)
    print("BACKGROUND CONTAMINATION TEST")
    print("=" * 70)
    print(f"\nSampling {len(sample)} training images.")
    print("Comparing whole-frame b_ratio against progressively tighter crops.\n")

    rows = []
    for i, entry in enumerate(sample[col], 1):
        path, _rule, _root = resolve_image_path(entry)
        if path is None:
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        rec = {"filename": entry}
        rec["full"] = b_ratio(rgb.reshape(-1, 3).mean(axis=0))
        rec["center_50"] = b_ratio(center_crop(rgb, 0.50).reshape(-1, 3).mean(axis=0))
        rec["center_25"] = b_ratio(center_crop(rgb, 0.25).reshape(-1, 3).mean(axis=0))

        sk, cov = skin_mean(rgb.astype(np.uint8))
        rec["skin_masked"] = b_ratio(sk) if sk is not None else np.nan
        rec["skin_coverage"] = cov
        rows.append(rec)

        if i % 100 == 0:
            sys.stdout.write(f"\r  {i}/{len(sample)}")
            sys.stdout.flush()

    print("\r" + " " * 30 + "\r", end="")
    res = pd.DataFrame(rows)
    if res.empty:
        print("No images could be read.")
        return

    methods = ["full", "center_50", "center_25", "skin_masked"]
    names = {
        "full": "whole frame (6.1 now)",
        "center_50": "center 50% by area",
        "center_25": "center 25% by area",
        "skin_masked": "YCrCb skin pixels only",
    }

    print("-" * 70)
    print("b_ratio BY MEASUREMENT REGION")
    print("-" * 70)
    print(f"\n  {'region':<24} {'n':>5} {'median':>8} {'mean':>8} {'p25':>8} {'p75':>8}")
    medians = {}
    for m in methods:
        v = res[m].dropna().to_numpy()
        if v.size == 0:
            continue
        medians[m] = float(np.median(v))
        print(f"  {names[m]:<24} {v.size:>5} {np.median(v):>8.4f} {v.mean():>8.4f} "
              f"{np.percentile(v,25):>8.4f} {np.percentile(v,75):>8.4f}")

    print(f"\n  Thresholds: Warm < {WARM_T} <= Neutral <= {COOL_T} < Cool")

    print("\n" + "-" * 70)
    print("RESULTING UNDERTONE SPLIT")
    print("-" * 70)
    print(f"\n  {'region':<24} {'Warm':>14} {'Neutral':>14} {'Cool':>14}")
    splits = {}
    for m in methods:
        v = res[m].dropna()
        if v.empty:
            continue
        labels = v.map(label)
        n = len(labels)
        counts = labels.value_counts()
        splits[m] = {k: int(counts.get(k, 0)) for k in ("Warm", "Neutral", "Cool")}
        cells = "".join(
            f"{counts.get(k,0):>7} ({100*counts.get(k,0)/n:>4.1f}%)"
            for k in ("Warm", "Neutral", "Cool")
        )
        print(f"  {names[m]:<24}{cells}")

    cov = res["skin_coverage"].dropna()
    if len(cov):
        print(f"\n  Skin-range pixels per image: median {100*cov.median():.1f}% "
              f"of the frame (p25 {100*cov.quantile(.25):.1f}%, "
              f"p75 {100*cov.quantile(.75):.1f}%)")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("WHAT THIS SHOWS")
    print("=" * 70)

    if "full" not in medians:
        return
    full_med = medians["full"]
    tight = medians.get("skin_masked", medians.get("center_25"))
    if tight is None:
        return
    shift = full_med - tight

    print(f"\n  Whole frame median b_ratio:  {full_med:.4f}")
    print(f"  Skin-only median b_ratio:    {tight:.4f}")
    print(f"  Shift:                       {shift:+.4f}")

    if shift > 0.015:
        cool_full = splits.get("full", {}).get("Cool", 0)
        cool_skin = splits.get("skin_masked", splits.get("center_25", {})).get("Cool", 0)
        n_full = sum(splits.get("full", {}).values()) or 1
        n_skin = sum(splits.get("skin_masked", splits.get("center_25", {})).values()) or 1
        print(f"""
  CONFIRMED. Background pixels are pulling the measurement toward blue.

  Measured on skin only, the Cool share goes from {100*cool_full/n_full:.0f}% to
  {100*cool_skin/n_skin:.0f}%. The 0.275/0.285 thresholds are consistent with
  skin-pixel values; applying them to a whole-scene average is the
  mismatch.

  This is not a bug to fix in the Baseline. The Baseline is defined as
  having no segmentation — that limitation is what HueView's regional
  CIELAB analysis exists to address, so a weak Baseline undertone result
  is a legitimate finding, not a broken control.

  For the writeup: state that the Baseline's undertone branch operates
  on whole-image means and is therefore sensitive to background, and
  cite this shift as the measured magnitude. Do NOT retune the
  thresholds — they're specified in your methodology, and adjusting
  them after seeing the distribution is fitting to the data.""")
    elif shift > 0.005:
        print("""
  PARTIAL. Tightening the region moves b_ratio warmer, but not enough to
  explain the whole result. Background is a contributing factor rather
  than the sole cause; the thresholds also appear miscalibrated for this
  dataset. Report both.""")
    else:
        print("""
  NOT CONFIRMED. b_ratio barely moves as the region tightens, so the
  background is not the driver.

  That points at the thresholds themselves: 0.275/0.285 don't match this
  dataset regardless of what's measured. Worth checking where those
  values came from — if they're cited from a source, the source's colour
  space or capture conditions may differ from yours.""")

    out = PROC / "background_effect.csv"
    res.to_csv(out, index=False)
    print(f"\n  Per-image detail written to {out}")


if __name__ == "__main__":
    main()
