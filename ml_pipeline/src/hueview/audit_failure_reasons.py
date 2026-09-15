"""
Diagnose exactly why 2,772 of 43,221 images have no usable HueView data --
specifically, whether the SCC-2 dropout found in audit_dropout_rates.py is
a file-availability problem (files only exist on a teammate's machine/
Drive, unrelated to any model behavior) or a genuine MediaPipe face-mesh
detection failure (a real, harder-to-fix limitation of the detector
itself).

Joins landmark_failures.csv (filename, reason) against the combined
train/val/test manifests (filename, SCC_label, illumination_label) to
answer: for each failure reason, what's its SCC_label / illumination_label
breakdown? This determines the actual remediation path -- getting missing
files vs. investigating detector behavior are not the same fix.

Usage:
    python audit_failure_reasons.py
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

FAILURES_PATH = PROJECT_ROOT / "data" / "processed" / "landmark_failures.csv"
SPLIT_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "train.csv",
    PROJECT_ROOT / "data" / "processed" / "val.csv",
    PROJECT_ROOT / "data" / "processed" / "test.csv",
]


def main():
    if not FAILURES_PATH.is_file():
        raise SystemExit(f"ERROR: {FAILURES_PATH} not found.")
    for p in SPLIT_PATHS:
        if not p.is_file():
            raise SystemExit(f"ERROR: {p} not found.")

    failures = pd.read_csv(FAILURES_PATH)
    print(f"{FAILURES_PATH.name}: {len(failures)} total failures")

    print("\n" + "=" * 70)
    print("FAILURE REASON COUNTS (overall)")
    print("=" * 70)
    print(failures["reason"].value_counts().to_string())

    manifest = pd.concat([pd.read_csv(p) for p in SPLIT_PATHS], ignore_index=True)
    merged = failures.merge(
        manifest[["filename", "SCC_label", "illumination_label"]], on="filename", how="left"
    )

    n_unmatched = int(merged["SCC_label"].isna().sum())
    if n_unmatched:
        print(f"\n[warn] {n_unmatched} failed filenames not found in train/val/test.csv "
              f"(check these are the same manifest landmark_extraction.py read)")

    print("\n" + "=" * 70)
    print("FAILURE REASON x SCC_label")
    print("=" * 70)
    print(pd.crosstab(merged["SCC_label"], merged["reason"]).to_string())

    print("\n" + "=" * 70)
    print("FAILURE REASON x illumination_label")
    print("=" * 70)
    print(pd.crosstab(merged["illumination_label"], merged["reason"]).to_string())

    print("\nWHAT TO LOOK FOR")
    print("  - if SCC-2's failures are mostly 'file_not_found', this is a data-")
    print("    availability gap (files exist only on a teammate's machine/Drive),")
    print("    not a detection bias -- fixable by re-running landmark_extraction.py")
    print("    wherever the full dataset is mounted, not by changing any code here.")
    print("  - if SCC-2's failures are mostly 'no_face_landmarks' (file exists")
    print("    locally, MediaPipe just couldn't find a face), that's a real")
    print("    detector limitation -- worth investigating threshold changes or")
    print("    accepting and documenting as a limitation, not something a re-run")
    print("    alone would fix.")


if __name__ == "__main__":
    main()
