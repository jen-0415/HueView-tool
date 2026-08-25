"""
generate_splits.py

Creates the frozen train/val/test manifest splits, stratified on SCC_label
(70% train / 20% val / 10% test), plus the train-only image augmentation
function.

Run once, from the project root, to create:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv

These three files are then FROZEN: every downstream pipeline (baseline and
HueView) must read from them directly. The script will refuse to overwrite
existing split files unless --force is passed.

Usage:
    python src/preprocessing/generate_splits.py
    python src/preprocessing/generate_splits.py --check-only   # re-verify an existing frozen split, no regeneration
    python src/preprocessing/generate_splits.py --force        # deliberately regenerate (breaks the freeze)
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from PIL import Image, ImageEnhance
except ImportError:  # Pillow missing -- augment_image() will raise a clear error if actually called
    Image = None
    ImageEnhance = None

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Generate frozen, SCC_label-stratified train/val/test splits.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"),
                         help="Path to the labeled manifest CSV (must already contain SCC_label).")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"),
                         help="Directory to write train.csv / val.csv / test.csv into.")
    parser.add_argument("--filename-column", type=str, default="filename",
                         help="Column that uniquely identifies each image (default: filename).")
    parser.add_argument("--label-column", type=str, default="SCC_label",
                         help="Column to stratify the split on (default: SCC_label).")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite existing split files. Breaks the freeze -- use deliberately.")
    parser.add_argument("--check-only", action="store_true",
                         help="Skip generation; just re-run the leakage check against existing split files.")
    return parser.parse_args()


def check_no_leakage(train_df, val_df, test_df, filename_col):
    """Exits with an error if any filename appears in more than one split."""
    train_files = set(train_df[filename_col])
    val_files = set(val_df[filename_col])
    test_files = set(test_df[filename_col])

    overlaps = {
        "train/val": train_files & val_files,
        "train/test": train_files & test_files,
        "val/test": val_files & test_files,
    }
    leaks = {pair: files for pair, files in overlaps.items() if files}
    if leaks:
        print("LEAKAGE CHECK FAILED:", file=sys.stderr)
        for pair, files in leaks.items():
            sample = sorted(files)[:5]
            print(f"  {pair}: {len(files)} overlapping filenames, e.g. {sample}", file=sys.stderr)
        sys.exit(1)

    total = len(train_files) + len(val_files) + len(test_files)
    print(f"OK: no filename appears in more than one split ({total} unique filenames across all three).")


def generate_splits(df, filename_col, label_col):
    """70/20/10 stratified split on label_col. First splits off 70% train,
    then divides the remaining 30% into 20% val / 10% test (test_size=1/3
    of that remainder)."""
    if df[filename_col].duplicated().any():
        dupes = df.loc[df[filename_col].duplicated(), filename_col].unique()[:5]
        sys.exit(f"ERROR: manifest has duplicate filenames before splitting, e.g. {list(dupes)}. Fix the manifest first.")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[label_col], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=1 / 3, stratify=temp_df[label_col], random_state=SEED
    )
    return train_df, val_df, test_df


def augment_image(
    image,
    hflip_prob: float = 0.5,
    max_rotation_deg: float = 15.0,
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
):
    """
    Random horizontal flip + rotation + color jitter on a single PIL image.

    Call this ONLY inside the training-time image-loading step, for images
    listed in train.csv (e.g. inside a Dataset.__getitem__ / tf.data .map()
    step) -- never on val/test images, and never before or during split
    generation. Augmenting before the split risks near-duplicate images
    leaking across train/val/test.
    """
    if Image is None:
        raise ImportError("Pillow is required for augment_image(). Install with: pip install Pillow")

    if random.random() < hflip_prob:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    angle = random.uniform(-max_rotation_deg, max_rotation_deg)
    image = image.rotate(angle, resample=Image.BILINEAR, expand=False)

    image = ImageEnhance.Brightness(image).enhance(random.uniform(*brightness_range))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(*contrast_range))
    image = ImageEnhance.Color(image).enhance(random.uniform(*saturation_range))

    return image


def main():
    args = parse_args()
    out_dir = args.out_dir
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    if args.check_only:
        missing = [p for p in (train_path, val_path, test_path) if not p.exists()]
        if missing:
            sys.exit(f"ERROR: missing split file(s): {[str(p) for p in missing]} -- nothing to check.")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        check_no_leakage(train_df, val_df, test_df, args.filename_column)
        return

    existing = [p for p in (train_path, val_path, test_path) if p.exists()]
    if existing and not args.force:
        names = ", ".join(p.name for p in existing)
        sys.exit(
            f"ERROR: {names} already exist in {out_dir} -- the split is frozen and both pipelines "
            f"must read the same three files. Pass --force to regenerate deliberately."
        )

    if not args.manifest.exists():
        sys.exit(f"ERROR: manifest not found at {args.manifest.resolve()}")

    df = pd.read_csv(args.manifest)

    for col in (args.filename_column, args.label_column):
        if col not in df.columns:
            sys.exit(f"ERROR: column '{col}' not found in {args.manifest}. Available columns: {list(df.columns)}")

    if df[args.label_column].isna().any():
        sys.exit(
            f"ERROR: {int(df[args.label_column].isna().sum())} rows have a null {args.label_column}. "
            f"Run add_scc_labels.py first and resolve any gaps before splitting."
        )

    train_df, val_df, test_df = generate_splits(df, args.filename_column, args.label_column)
    check_no_leakage(train_df, val_df, test_df, args.filename_column)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    n = len(df)
    print(f"\nWrote frozen splits to {out_dir.resolve()}:")
    print(f"  train.csv: {len(train_df):5d} rows ({len(train_df) / n:.1%})")
    print(f"  val.csv:   {len(val_df):5d} rows ({len(val_df) / n:.1%})")
    print(f"  test.csv:  {len(test_df):5d} rows ({len(test_df) / n:.1%})")
    print("\nReminder: augment_image() must only be applied when loading images from")
    print("train.csv, at training time -- never to val/test, never before this split.")


if __name__ == "__main__":
    main()