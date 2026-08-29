"""
Characterize how the roots differ
=================================

verify_root_choice.py turned up a contradiction worth resolving:

  - All three main roots reproduce the stored mean_Y with median
    difference 0.000 (identical luminance)
  - Yet the same files differ by ~35/255 per pixel on average

Identical mean luminance plus large pixel difference is the exact
signature of a geometric transform. A horizontal flip changes nearly
every pixel while preserving the mean of every channel perfectly.

If the roots hold flipped/rotated copies, they're augmentation output,
not competing preprocessing runs — and that changes what you do about
them completely. This script tests each hypothesis directly.

Run from the repo root:  python src/baseline/characterize_roots.py

Read-only.
"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from path_resolver import find_all_matches, DEFAULT_ROOTS, V5_ROOT
except ImportError:
    from src.baseline.path_resolver import find_all_matches, DEFAULT_ROOTS, V5_ROOT

PROC = Path("data/processed")
MANIFEST = PROC / "manifest.csv"
SAMPLE = 120
PIX_TOL = 3.0  # mean abs diff below this counts as "the same image"


def load(path):
    img = cv2.imread(str(path))
    return img


def relationship(a, b):
    """
    Classify how image b relates to image a.
    Returns a label plus the mean absolute difference for the best fit.
    """
    if a.shape != b.shape:
        # Try matching after resizing b to a's dimensions
        try:
            b_rs = cv2.resize(b, (a.shape[1], a.shape[0]))
            d = float(np.abs(a.astype(np.int16) - b_rs.astype(np.int16)).mean())
            if d <= PIX_TOL:
                return "resized_copy", d
        except cv2.error:
            pass
        return "different_dimensions", float("nan")

    def diff(x, y):
        return float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean())

    tests = {
        "identical": b,
        "h_flip": cv2.flip(b, 1),
        "v_flip": cv2.flip(b, 0),
        "rot180": cv2.rotate(b, cv2.ROTATE_180),
        "rot90_cw": None,
        "rot90_ccw": None,
    }
    if a.shape[0] == a.shape[1]:
        tests["rot90_cw"] = cv2.rotate(b, cv2.ROTATE_90_CLOCKWISE)
        tests["rot90_ccw"] = cv2.rotate(b, cv2.ROTATE_90_COUNTERCLOCKWISE)

    best_label, best_d = None, float("inf")
    for label, cand in tests.items():
        if cand is None or cand.shape != a.shape:
            continue
        d = diff(a, cand)
        if d < best_d:
            best_label, best_d = label, d

    if best_d <= PIX_TOL:
        return best_label, best_d

    # Not a geometric transform. Is it a brightness/contrast shift?
    a_m, b_m = a.astype(np.float32).mean(), b.astype(np.float32).mean()
    if abs(a_m - b_m) > 5:
        return "brightness_shift", diff(a, b)
    return "unrelated", diff(a, b)


def mean_y(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].mean())


def main():
    if not MANIFEST.exists():
        print(f"!! {MANIFEST} not found.")
        return

    manifest = pd.read_csv(MANIFEST)
    col = "filename" if "filename" in manifest.columns else manifest.columns[0]
    y_col = next((c for c in manifest.columns
                  if c.lower() in ("mean_y", "meany", "y_mean", "luminance")), None)

    roots = [Path(r) for r in DEFAULT_ROOTS] + [Path(V5_ROOT)]

    print("=" * 70)
    print("HOW DO THE ROOTS DIFFER?")
    print("=" * 70)
    print(f"\nSampling up to {SAMPLE} files present under more than one root...\n")

    rels = Counter()
    pairs_seen = Counter()
    examples = {}
    y_agree = 0
    y_total = 0
    y_mismatch_examples = []

    checked = 0
    for _, row in manifest.sample(frac=1.0, random_state=23).iterrows():
        matches = find_all_matches(row[col])
        if len(matches) < 2:
            continue

        # Identify which root each match came from
        tagged = []
        for m in matches:
            for r in roots:
                try:
                    m.relative_to(r)
                    tagged.append((r.name, m))
                    break
                except ValueError:
                    continue
        if len(tagged) < 2:
            continue

        (r1, p1), (r2, p2) = tagged[0], tagged[1]
        a, b = load(p1), load(p2)
        if a is None or b is None:
            continue

        label, d = relationship(a, b)
        rels[label] += 1
        pairs_seen[f"{r1} vs {r2}"] += 1
        if label not in examples:
            examples[label] = (row[col], r1, r2, d, a.shape, b.shape)

        # Does the stored mean_Y agree with both versions?
        if y_col and not pd.isna(row[y_col]):
            y_total += 1
            ya, yb = mean_y(a), mean_y(b)
            stored = float(row[y_col])
            if abs(ya - stored) <= 0.5 and abs(yb - stored) <= 0.5:
                y_agree += 1
            elif len(y_mismatch_examples) < 3:
                y_mismatch_examples.append((row[col], stored, ya, yb, r1, r2))

        checked += 1
        if checked >= SAMPLE:
            break

    if checked == 0:
        print("  No multi-root files found.")
        return

    print("-" * 70)
    print("RELATIONSHIP BETWEEN VERSIONS")
    print("-" * 70)
    print(f"\n  {'relationship':<24} {'count':>7} {'share':>8}")
    for label, n in rels.most_common():
        print(f"  {label:<24} {n:>7} {100*n/checked:>7.1f}%")

    print("\n  Example of each:")
    for label, (fn, r1, r2, d, s1, s2) in examples.items():
        print(f"    {label}: {fn}")
        print(f"        {r1} {s1}  vs  {r2} {s2}   mean|diff| = {d:.2f}")

    print("\n  Root pairs sampled:")
    for pair, n in pairs_seen.most_common():
        print(f"    {pair}: {n}")

    if y_col and y_total:
        print("\n" + "-" * 70)
        print("STORED mean_Y vs BOTH VERSIONS")
        print("-" * 70)
        print(f"\n  Both versions reproduce stored mean_Y: {y_agree}/{y_total} "
              f"({100*y_agree/y_total:.1f}%)")
        if y_mismatch_examples:
            print("\n  Examples where they don't:")
            for fn, st, ya, yb, r1, r2 in y_mismatch_examples:
                print(f"    {fn}")
                print(f"        stored={st:.3f}   {r1}={ya:.3f}   {r2}={yb:.3f}")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("WHAT THIS MEANS")
    print("=" * 70)

    geometric = sum(rels[k] for k in ("h_flip", "v_flip", "rot180", "rot90_cw", "rot90_ccw"))
    same = rels["identical"] + rels["resized_copy"]
    unrelated = rels["unrelated"] + rels["brightness_shift"] + rels["different_dimensions"]

    if same / checked > 0.8:
        print("\n  The roots hold the SAME images. The byte differences are just")
        print("  re-encoding. Any root works; search order doesn't matter.")
        print("  Proceed with `processed` and move on.")
    elif geometric / checked > 0.5:
        print(f"\n  {100*geometric/checked:.0f}% of pairs are FLIPPED OR ROTATED copies.")
        print("""
  These folders are augmentation output, not competing preprocessing
  runs. That explains the contradiction: a flip preserves mean_Y exactly
  while changing nearly every pixel.

  This is a problem, and a different one than we thought. If augmented
  copies of the same face carry the same filename in different folders,
  and your splits reference that filename, then which augmentation you
  train on depends on folder search order.

  Worse: Phase 5 specifies augmentation applies to the TRAINING SET ONLY,
  after the split. If augmented copies are sitting in folders that val
  and test rows also resolve against, augmented images could be entering
  your evaluation set — which the manuscript explicitly rules out.

  Confirm with whoever produced these folders what c2_processed and
  'processed - 7-26' actually are before training anything.""")
    else:
        print(f"\n  Mixed picture: {100*same/checked:.0f}% same, "
              f"{100*geometric/checked:.0f}% geometric transforms, "
              f"{100*unrelated/checked:.0f}% genuinely different.")
        print("""
  No single clean story. These folders were likely produced at different
  times by different scripts, with partial overlap.

  Given `processed` is the most complete set (all 10 MST folders, 30,735
  files) and is already first in search order, the pragmatic path is to
  restrict resolution to `processed` + `v5_processed` only, dropping the
  other two roots from the search entirely. That makes which-image-you-get
  deterministic rather than dependent on folder ordering.

  Check first how many rows would stop resolving under that restriction —
  if it's only the 432 that currently hit c2_processed, that's a small,
  documentable exclusion.""")


if __name__ == "__main__":
    main()
