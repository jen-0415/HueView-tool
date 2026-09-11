"""
Phase 8/10 preparation -- audit the real, usable HueView training set.

landmark_extraction.py (7.1) succeeded on 40,449 of the full 43,221-image
Phase 5 manifest -- everything downstream (faces_ssr/, regions/,
label_maps/) only exists for that 40,449. Baseline (Phase 6) can train and
evaluate on the full three splits at their stated size, since it only
needs MTCNN's face detection, not landmarks. HueView cannot: there is no
label map to decode for an image with no landmarks.

This script reports the real, usable HueView count per split, and
re-checks the SCC/illumination stratification Phase 5's own checkpoint
calls for ("SCC proportions roughly match across splits") on that filtered
subset specifically. Dropping ~6.4% of images is only harmless if the drop
is roughly even across classes -- if landmark failure correlates with
something like skin tone or lighting, that is a fairness-relevant finding
worth surfacing here, not silently absorbing into a smaller denominator.

Usage:
    python audit_hueview_splits.py
"""

import sys
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------ project root --


def find_project_root(start: Path, marker: str = "data") -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)

SPLIT_PATHS = {
    "train": PROJECT_ROOT / "data" / "processed" / "train.csv",
    "val": PROJECT_ROOT / "data" / "processed" / "val.csv",
    "test": PROJECT_ROOT / "data" / "processed" / "test.csv",
}
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def pct_table(df: pd.DataFrame, col: str) -> pd.Series:
    return (df[col].value_counts(normalize=True) * 100).round(2).sort_index()


def main():
    for name, path in SPLIT_PATHS.items():
        if not path.is_file():
            raise SystemExit(f"ERROR: {path} not found.")
    if not INDEX_PATH.is_file():
        raise SystemExit(f"ERROR: {INDEX_PATH} not found -- run landmark_extraction.py (7.1) first.")

    usable_filenames = set(pd.read_csv(INDEX_PATH)["filename"])
    print(f"landmarks_index.csv: {len(usable_filenames)} filenames with usable HueView data\n")

    print("=" * 78)
    print("SPLIT SIZES: FULL (BASELINE-USABLE) vs HUEVIEW-USABLE")
    print("=" * 78)

    usable_dfs = {}
    for name, path in SPLIT_PATHS.items():
        df = pd.read_csv(path)
        usable_df = df[df["filename"].isin(usable_filenames)].copy()
        usable_dfs[name] = usable_df

        n_full = len(df)
        n_usable = len(usable_df)
        drop_rate = 100 * (1 - n_usable / n_full)
        print(f"{name:6s}  full={n_full:6d}   hueview_usable={n_usable:6d}   dropped={n_full - n_usable:5d} ({drop_rate:.2f}%)")

    print("\n" + "=" * 78)
    print("SCC_label PROPORTIONS: FULL SPLIT vs HUEVIEW-USABLE SUBSET")
    print("=" * 78)
    for name, path in SPLIT_PATHS.items():
        df = pd.read_csv(path)
        usable_df = usable_dfs[name]

        full_pct = pct_table(df, "SCC_label")
        usable_pct = pct_table(usable_df, "SCC_label")
        compare = pd.DataFrame({"full_%": full_pct, "usable_%": usable_pct})
        compare["diff_pts"] = (compare["usable_%"] - compare["full_%"]).round(2)

        print(f"\n[{name}]")
        print(compare.to_string())
        max_shift = compare["diff_pts"].abs().max()
        if max_shift >= 2.0:
            print(f"  ** flag: max shift {max_shift:.2f} points -- worth a closer look **")

    print("\n" + "=" * 78)
    print("illumination_label PROPORTIONS: FULL SPLIT vs HUEVIEW-USABLE SUBSET")
    print("=" * 78)
    for name, path in SPLIT_PATHS.items():
        df = pd.read_csv(path)
        usable_df = usable_dfs[name]

        full_pct = pct_table(df, "illumination_label")
        usable_pct = pct_table(usable_df, "illumination_label")
        compare = pd.DataFrame({"full_%": full_pct, "usable_%": usable_pct})
        compare["diff_pts"] = (compare["usable_%"] - compare["full_%"]).round(2)

        print(f"\n[{name}]")
        print(compare.to_string())
        max_shift = compare["diff_pts"].abs().max()
        if max_shift >= 2.0:
            print(f"  ** flag: max shift {max_shift:.2f} points -- worth a closer look **")

    # Save the filtered, HueView-usable manifests for Phase 10 to read directly.
    for name, usable_df in usable_dfs.items():
        out_path = OUT_DIR / f"{name}_hueview_usable.csv"
        usable_df.to_csv(out_path, index=False)
        print(f"\nSaved {out_path} ({len(usable_df)} rows)")


if __name__ == "__main__":
    main()
