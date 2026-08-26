"""
add_scc_labels.py

Adds an SCC (Skin Color Classification) label to every row of the processed
manifest by mapping the existing MST (Monk Skin Tone) label through the
fixed MST -> SCC lookup table. Refuses to write anything unless every row
ends up with a valid SCC-1..SCC-6 label, since SCC_label is the ground
truth the whole study is evaluated against.

Usage (run from the project root):
    python src/preprocessing/add_scc_labels.py
    python src/preprocessing/add_scc_labels.py --mst-column MST --output data/processed/manifest_labeled.csv
"""

import argparse
import shutil
import sys
from pathlib import Path
    
import pandas as pd

MST_TO_SCC = {
    "MST-1": "SCC-1",
    "MST-2": "SCC-1",
    "MST-3": "SCC-2",
    "MST-4": "SCC-2",
    "MST-5": "SCC-3",
    "MST-6": "SCC-3",
    "MST-7": "SCC-4",
    "MST-8": "SCC-5",
    "MST-9": "SCC-5",
    "MST-10": "SCC-6",
}

VALID_SCC_LABELS = {f"SCC-{i}" for i in range(1, 7)}


def parse_args():
    parser = argparse.ArgumentParser(description="Add SCC labels to manifest.csv via the MST -> SCC mapping.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"),
                         help="Path to the manifest CSV (default: data/processed/manifest.csv)")
    parser.add_argument("--mst-column", type=str, default="mst_label",
                         help="Existing column holding MST labels (default: mst_label)")
    parser.add_argument("--scc-column", type=str, default="SCC_label",
                         help="New column to write SCC labels to (default: SCC_label)")
    parser.add_argument("--output", type=Path, default=None,
                         help="Where to write the updated manifest. Defaults to overwriting --manifest in place.")
    parser.add_argument("--no-backup", action="store_true",
                         help="Skip writing a .bak copy of the manifest before an in-place overwrite.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.manifest
    output_path = args.output or manifest_path
    in_place = output_path == manifest_path

    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found at {manifest_path.resolve()}")

    df = pd.read_csv(manifest_path)

    if args.mst_column not in df.columns:
        sys.exit(
            f"ERROR: expected MST column '{args.mst_column}' not found in {manifest_path}.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Pass --mst-column <name> if it's called something else."
        )

    # Map MST -> SCC. Anything not in the lookup table (typos, unexpected
    # labels, blanks) becomes NaN instead of silently propagating.
    df[args.scc_column] = df[args.mst_column].map(MST_TO_SCC)

    # --- Ground-truth validation ---------------------------------------
    # Every row must end up with a non-null, valid SCC-1..SCC-6 label.
    # This is the only ground truth the study is evaluated against, so a
    # gap here is fatal -- the script exits without writing anything.
    null_mask = df[args.scc_column].isna()
    invalid_mask = ~null_mask & ~df[args.scc_column].isin(VALID_SCC_LABELS)

    if null_mask.any() or invalid_mask.any():
        bad_mst_values = sorted(df.loc[null_mask, args.mst_column].dropna().unique().tolist())
        print("VALIDATION FAILED: not every row has a valid SCC-1..SCC-6 label.", file=sys.stderr)
        print(f"  Rows with no SCC mapping (null):  {int(null_mask.sum())}", file=sys.stderr)
        print(f"  Rows with an invalid SCC value:   {int(invalid_mask.sum())}", file=sys.stderr)
        if bad_mst_values:
            print(f"  MST values with no entry in MST_TO_SCC: {bad_mst_values}", file=sys.stderr)
        print("\n  First 20 offending rows:", file=sys.stderr)
        cols_to_show = [c for c in (args.mst_column, args.scc_column) if c in df.columns]
        print(df.loc[null_mask | invalid_mask, cols_to_show].head(20).to_string(), file=sys.stderr)
        sys.exit(1)

    if in_place and not args.no_backup:
        backup_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(manifest_path, backup_path)
            print(f"Backed up original manifest to {backup_path.name}")

    df.to_csv(output_path, index=False)
    print(f"OK: all {len(df)} rows have a valid SCC-1..SCC-6 label.")
    print(f"Wrote labeled manifest to {output_path.resolve()}")


if __name__ == "__main__":
    main()