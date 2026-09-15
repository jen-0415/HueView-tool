"""
One-time full verification of the HueView dataset.

Checks, for every row in train/val/test, whether all SIX required image files
exist on disk:
    - 5 regional patches: data/processed/regions/<MST>/<stem>_<region>.jpg
    - 1 full face:        data/processed/images/processed/<possible locations>

Rows missing even one file are dropped. This catches the Windows
"(2)" duplicate-renaming problem in one pass instead of one FileNotFoundError
at a time.

Run:
    python3 verify_splits.py

Produces:
    train_verified.csv / val_verified.csv / test_verified.csv
    missing_files_report.csv   <- exactly which file was missing per dropped row
"""
import pandas as pd
import pathlib

ROOT = pathlib.Path("/mnt/c/Users/Elizabeth/HueView-tool")
SPLIT_DIR = ROOT / "data/processed"
REGIONS_DIR = SPLIT_DIR / "regions"
FACE_DIR = SPLIT_DIR / "images/processed"
REGIONS = ["forehead", "left_cheek", "right_cheek", "jawline", "nose_bridge"]

SOURCE_CSVS = {
    "train": "train_hueview_usable.csv",
    "val": "val_hueview_usable.csv",
    "test": "test_hueview_usable.csv",
}


FACE_SOURCE_DIRS = ["processed", "c1_processed", "c2_processed", "v5_processed"]


def find_face_file(fname: str):
    """Full faces are split across four source folders. Search all four,
    with a fallback for Windows '(2)' duplicate-renamed filenames."""
    parts = pathlib.Path(fname)
    stem = parts.stem
    folder = parts.parent.name

    stems = [stem]

    # Handle duplicate-renamed filenames such as:
    # 0008_1_0_0_01 (2).jpg -> 0008_1_0_0_01.jpg
    if stem.endswith(" (2)"):
        stems.append(stem[:-4])

    for src in FACE_SOURCE_DIRS:
        for current_stem in stems:
            for ext in (".jpg", ".png"):
                c = SPLIT_DIR / "images" / src / folder / f"{current_stem}{ext}"
                if c.exists():
                    return c

    return None


def find_region_file(fname: str, region: str):
    parts = pathlib.Path(fname)
    stem = parts.stem
    folder = parts.parent.name

    candidates = [
        REGIONS_DIR / folder / f"{stem}_{region}.jpg",
    ]

    if stem.endswith(" (2)"):
        stem_no_dup = stem[:-4]
        candidates.append(
            REGIONS_DIR / folder / f"{stem_no_dup}_{region}.jpg"
        )

    for path in candidates:
        if path.exists():
            return path

    return None


def check_row(fname: str):
    """Returns a list of missing file descriptions; empty list = all present."""
    missing = []
    for region in REGIONS:
        if find_region_file(fname, region) is None:
            missing.append(f"region:{region}")
    if find_face_file(fname) is None:
        missing.append("full_face")
    return missing


def main():
    report_rows = []

    for split, source in SOURCE_CSVS.items():
        path = SPLIT_DIR / source
        if not path.exists():
            print(f"[skip] {source} not found at {path}")
            continue

        df = pd.read_csv(path)
        keep_mask = []
        for fname in df["filename"]:
            missing = check_row(fname)
            keep_mask.append(len(missing) == 0)
            if missing:
                report_rows.append({"split": split, "filename": fname,
                                    "missing": ";".join(missing)})

        before = len(df)
        df_keep = df[pd.Series(keep_mask, index=df.index)]
        after = len(df_keep)
        print(f"{split}: {before} -> {after} ({before - after} dropped)")
        df_keep.to_csv(SPLIT_DIR / f"{split}_verified.csv", index=False)

    if report_rows:
        rep = pd.DataFrame(report_rows)
        rep.to_csv(SPLIT_DIR / "missing_files_report.csv", index=False)
        print(f"\nWrote missing_files_report.csv ({len(rep)} problem rows)")
        print("\nMost common missing-file types:")
        print(rep["missing"].value_counts().head(10).to_string())
    else:
        print("\nNo missing files at all — every row is fully usable.")


if __name__ == "__main__":
    main()