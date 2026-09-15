"""
Follow-up to audit_hueview_splits.py -- exact per-class landmark-detection
dropout rates.

The first pass (proportion-of-total comparison between the full split and
the HueView-usable subset) flagged a large, consistent shift for SCC-2
across all three splits, plus a smaller consistent shift for Low
illumination. Comparing proportions computed against two different
denominators (full split size vs. usable subset size) can make an effect
look bigger or smaller than it actually is -- this script instead computes
the dropout rate per class directly: what fraction of each class's own
images have no usable landmarks, full stop. It also cross-tabulates
SCC_label x illumination_label to check whether the SCC-2 effect and the
Low-illumination effect are the same underlying cause (e.g. SCC-2 happens
to be photographed under Low illumination more often in this dataset) or
two independent effects.

Usage:
    python audit_dropout_rates.py
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


def dropout_table(df: pd.DataFrame, usable_filenames: set, group_cols: list) -> pd.DataFrame:
    df = df.copy()
    df["usable"] = df["filename"].isin(usable_filenames)
    grouped = df.groupby(group_cols)["usable"].agg(full_count="count", usable_count="sum")
    grouped["dropped"] = grouped["full_count"] - grouped["usable_count"]
    grouped["dropout_rate_%"] = (100 * grouped["dropped"] / grouped["full_count"]).round(2)
    return grouped.sort_values("dropout_rate_%", ascending=False)


def main():
    for path in SPLIT_PATHS.values():
        if not path.is_file():
            raise SystemExit(f"ERROR: {path} not found.")
    if not INDEX_PATH.is_file():
        raise SystemExit(f"ERROR: {INDEX_PATH} not found.")

    usable_filenames = set(pd.read_csv(INDEX_PATH)["filename"])

    # Pool all three splits -- the point here is characterizing the
    # pipeline's failure pattern itself, not comparing splits to each other.
    all_rows = pd.concat([pd.read_csv(p) for p in SPLIT_PATHS.values()], ignore_index=True)
    overall_dropout = 100 * (1 - all_rows["filename"].isin(usable_filenames).mean())

    print(f"Pooled across train+val+test: {len(all_rows)} images, "
          f"overall dropout rate = {overall_dropout:.2f}%\n")

    print("=" * 78)
    print("DROPOUT RATE BY SCC_label (pooled, sorted worst first)")
    print("=" * 78)
    print(dropout_table(all_rows, usable_filenames, ["SCC_label"]).to_string())

    print("\n" + "=" * 78)
    print("DROPOUT RATE BY illumination_label (pooled, sorted worst first)")
    print("=" * 78)
    print(dropout_table(all_rows, usable_filenames, ["illumination_label"]).to_string())

    print("\n" + "=" * 78)
    print("DROPOUT RATE BY SCC_label x illumination_label (pooled, sorted worst first)")
    print("=" * 78)
    print(dropout_table(all_rows, usable_filenames, ["SCC_label", "illumination_label"]).to_string())

    print("\nWHAT TO LOOK FOR")
    print("  - if SCC-2's dropout rate is high across ALL illumination levels in the")
    print("    crosstab (not just Low), the effect is tied to skin tone independent")
    print("    of lighting.")
    print("  - if SCC-2's dropout rate is only elevated specifically under Low")
    print("    illumination (and closer to average under Medium/High), the two")
    print("    findings are likely the same underlying cause, not two separate ones.")


if __name__ == "__main__":
    main()
