"""
Verify which root Phase 3 actually used
=======================================

56% of files that exist under multiple roots have different bytes.
`processed` currently wins by search order. But Phase 3 computed mean_Y
and the illumination labels from ONE specific set, and if that wasn't
`processed`, every illumination label in the manifest describes pixels
that Phase 6 will never read.

The manifest stores mean_Y per image. That's a fingerprint. Recompute
mean_Y from each root and see which one reproduces the stored value.

Run from the repo root:  python src/baseline/verify_root_choice.py

Read-only.
"""

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
SAMPLE = 150
TOL = 0.5  # mean_Y units; tighter than any plausible rounding difference


def mean_y_from_rgb(path):
    """Phase 3's recipe: load, convert to YCrCb, mean of Y."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return float(ycrcb[:, :, 0].mean())


def mean_y_from_bgr(path):
    """Same, but if Phase 3 forgot the BGR->RGB step (a very common slip)."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return float(ycrcb[:, :, 0].mean())


def main():
    if not MANIFEST.exists():
        print(f"!! {MANIFEST} not found.")
        return

    manifest = pd.read_csv(MANIFEST)
    col = "filename" if "filename" in manifest.columns else manifest.columns[0]

    y_col = next((c for c in manifest.columns
                  if c.lower() in ("mean_y", "meany", "y_mean", "luminance")), None)
    if y_col is None:
        print("!! No mean_Y column in manifest.csv. Columns present:")
        print(f"   {list(manifest.columns)}")
        print("\n   Without a stored luminance value there's no fingerprint to match.")
        print("   Check step2_luminance.csv instead — if it has one, point this")
        print("   script at that file.")
        return

    print("=" * 70)
    print("WHICH ROOT DID PHASE 3 USE?")
    print("=" * 70)
    print(f"\nUsing manifest column '{y_col}' as the fingerprint.")
    print(f"Scanning for rows that exist under more than one root...")

    roots = [Path(r) for r in DEFAULT_ROOTS] + [Path(V5_ROOT)]
    root_names = [r.name for r in roots]

    # Collect rows present under 2+ roots
    candidates = []
    for _, row in manifest.sample(frac=1.0, random_state=11).iterrows():
        matches = find_all_matches(row[col])
        if len(matches) < 2:
            continue
        candidates.append((row[col], row[y_col], matches))
        if len(candidates) >= SAMPLE:
            break

    if not candidates:
        print("\n  No rows exist under more than one root. Nothing to disambiguate.")
        return

    print(f"Testing {len(candidates)} such rows.\n")

    # For each root, count how often its recomputed mean_Y matches the stored one
    hits_rgb = {n: 0 for n in root_names}
    hits_bgr = {n: 0 for n in root_names}
    tested = {n: 0 for n in root_names}
    deltas = {n: [] for n in root_names}

    for fname, stored, matches in candidates:
        if pd.isna(stored):
            continue
        for m in matches:
            rname = None
            for r in roots:
                try:
                    m.relative_to(r)
                    rname = r.name
                    break
                except ValueError:
                    continue
            if rname is None:
                continue

            v_rgb = mean_y_from_rgb(m)
            v_bgr = mean_y_from_bgr(m)
            if v_rgb is None:
                continue
            tested[rname] += 1
            deltas[rname].append(abs(v_rgb - float(stored)))
            if abs(v_rgb - float(stored)) <= TOL:
                hits_rgb[rname] += 1
            if v_bgr is not None and abs(v_bgr - float(stored)) <= TOL:
                hits_bgr[rname] += 1

    print("-" * 70)
    print("MATCH AGAINST STORED mean_Y")
    print("-" * 70)
    print(f"\n  {'root':<22} {'tested':>8} {'match':>8} {'rate':>8} {'median |diff|':>15}")
    for n in root_names:
        if tested[n] == 0:
            continue
        rate = 100 * hits_rgb[n] / tested[n]
        med = float(np.median(deltas[n])) if deltas[n] else float("nan")
        print(f"  {n:<22} {tested[n]:>8} {hits_rgb[n]:>8} {rate:>7.1f}% {med:>15.3f}")

    # BGR variant, in case Phase 3 skipped the colour conversion
    any_bgr = any(hits_bgr[n] > hits_rgb[n] for n in root_names if tested[n])
    if any_bgr:
        print("\n  (Alternative: mean_Y computed WITHOUT the BGR->RGB conversion)")
        print(f"  {'root':<22} {'tested':>8} {'match':>8} {'rate':>8}")
        for n in root_names:
            if tested[n] == 0:
                continue
            rate = 100 * hits_bgr[n] / tested[n]
            print(f"  {n:<22} {tested[n]:>8} {hits_bgr[n]:>8} {rate:>7.1f}%")

    # ---- How different are the roots, really? ----
    print("\n" + "=" * 70)
    print("HOW DIFFERENT ARE THE VERSIONS?")
    print("=" * 70)
    print("\nDifferent bytes can still mean visually identical images (a JPEG")
    print("re-encode). Comparing actual pixels tells you whether this matters.\n")

    diffs = []
    shapes = []
    for fname, stored, matches in candidates[:40]:
        imgs = []
        for m in matches[:2]:
            im = cv2.imread(str(m))
            if im is not None:
                imgs.append((m, im))
        if len(imgs) < 2:
            continue
        (p1, a), (p2, b) = imgs[0], imgs[1]
        if a.shape != b.shape:
            shapes.append((fname, a.shape, b.shape))
            continue
        diffs.append(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()))

    if shapes:
        print(f"  {len(shapes)} sampled pair(s) have DIFFERENT DIMENSIONS:")
        for fn, s1, s2 in shapes[:3]:
            print(f"      {fn}: {s1} vs {s2}")
        print("  Different dimensions means different cropping — these are not")
        print("  re-encodes, they're different preprocessing runs.\n")

    if diffs:
        arr = np.array(diffs)
        print(f"  Mean absolute pixel difference across {len(arr)} same-size pairs:")
        print(f"      median {np.median(arr):.2f}   mean {arr.mean():.2f}   max {arr.max():.2f}")
        print()
        if np.median(arr) < 2:
            print("  Small (<2/255). Consistent with JPEG re-encoding — the images")
            print("  are visually the same. Low risk either way.")
        elif np.median(arr) < 10:
            print("  Moderate. More than re-encoding noise. Worth confirming which")
            print("  version your preprocessing was validated against.")
        else:
            print("  LARGE. These are substantively different images — different")
            print("  crops, different enhancement, or different source photos.")
            print("  Reading the wrong root would change your results materially.")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    scored = [(hits_rgb[n] / tested[n], n) for n in root_names if tested[n] > 0]
    scored.sort(reverse=True)

    if not scored or scored[0][0] < 0.5:
        print("\n  No root reproduces the stored mean_Y values.")
        print("\n  That means Phase 3 computed luminance from images that are not")
        print("  in any of these four folders — an earlier, since-overwritten run.")
        print("  The illumination labels in your manifest cannot be verified")
        print("  against the pixels you'd train on.")
        print("\n  Safest fix: recompute mean_Y and the K-Means illumination bins")
        print("  from `processed` and rewrite those two manifest columns. The SCC")
        print("  labels are untouched by this — only illumination is affected.")
        print("  It re-runs Phase 3 step 2-4 and nothing downstream of it.")
    else:
        best_rate, best = scored[0]
        print(f"\n  Best match: {best}  ({100*best_rate:.1f}% of sampled rows)")
        if best_rate > 0.95:
            if best == "processed":
                print("\n  This is already first in path_resolver.DEFAULT_ROOTS, so")
                print("  Phase 6 will read the same pixels Phase 3 did. Nothing to change.")
            else:
                print(f"\n  !! Phase 3 used '{best}', but path_resolver searches")
                print("     'processed' first. Reorder DEFAULT_ROOTS so this root")
                print("     comes first, or your features and illumination labels")
                print("     will describe different images.")
        else:
            print("\n  Partial match only. Phase 3's images may have been a mix of")
            print("  roots, or reprocessed after luminance was computed. Recomputing")
            print("  mean_Y from a single root is the reliable way forward.")


if __name__ == "__main__":
    main()
