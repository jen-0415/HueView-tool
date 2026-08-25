"""
generate_splits.py

Creates the frozen train/val/test manifest splits, grouped and stratified
by PERSON (not by individual photo), so no one's face appears in more than
one split.

Person ID extraction handles three source-dataset naming conventions
confirmed in this manifest:
  - FairFace: one photo per identity already -- each row is its own person
  - LFW: "FirstName_LastName_LFW..." -- true ID is everything before "_LFW"
    (this is what correctly separates George W. Bush / George Clooney /
    George Lopez, previously merged into one fake "George")
  - Everything else: first underscore-delimited token (verified against
    normal-sized groups, e.g. the Faces94/95/96-style "smille" set)

Run once, from the project root:
    python src/preprocessing/generate_splits.py --force

(--force is required this run specifically: the existing train/val/test
CSVs were built by the old per-photo split and need replacing, not just
the code going forward.)

Usage:
    python src/preprocessing/generate_splits.py --force
    python src/preprocessing/generate_splits.py --check-only
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from PIL import Image, ImageEnhance
except ImportError:
    Image = None
    ImageEnhance = None

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Generate frozen, person-grouped, SCC_label-stratified train/val/test splits.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--filename-column", type=str, default="filename")
    parser.add_argument("--label-column", type=str, default="SCC_label")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite existing split files. Required this run: the existing "
                              "splits were built by the old, leaky per-photo version.")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def extract_person_id(df, filename_col):
    """Person ID extraction, handling FairFace / LFW / numbered-PNG / other
    naming conventions."""
    basename = df[filename_col].str.split("/").str[-1]

    is_fairface = basename.str.contains("FairFace", case=False, na=False)
    is_lfw = basename.str.contains("LFW", case=False, na=False) & ~is_fairface
    # A second source that reuses Brazilian Faces' plain numeric IDs --
    # pattern "<number>_<single digit 0-4>.png", e.g. "103_0.png". Without
    # this, "103_Brazilian Faces103-01_face_1.jpg" (Brazilian Faces) and
    # "103_0.png" (this other, unrelated source) collide into one fake ID.
    is_numbered_png = basename.str.match(r"^\d+_\d\.png$", case=False, na=False) & ~is_fairface & ~is_lfw
    other = ~is_fairface & ~is_lfw & ~is_numbered_png

    person_id = pd.Series(index=df.index, dtype=object)
    person_id[is_fairface] = "fairface_" + df.index[is_fairface].astype(str)
    person_id[is_lfw] = basename[is_lfw].str.split("_LFW").str[0]
    person_id[is_numbered_png] = "pngset_" + basename[is_numbered_png].str.extract(r"^(\d+)_")[0].values
    person_id[other] = basename[other].str.extract(r"^([^_]+)_")[0].values

    return person_id


def build_person_table(df, person_col, label_col):
    """One row per person, with their majority SCC label. A person whose
    photos disagree on SCC label is a red flag -- skin tone shouldn't vary
    per photo, so this usually means two different real people got merged
    under one extracted ID. Surfaced as a warning, not a hard failure,
    since it's worth a manual look rather than silently guessing."""
    grouped = df.groupby(person_col)[label_col]
    n_unique_labels = grouped.nunique()
    inconsistent = n_unique_labels[n_unique_labels > 1]

    if len(inconsistent) > 0:
        print(f"WARNING: {len(inconsistent)} person ID(s) have inconsistent {label_col} across their photos.")
        print("This usually means two different real people share one extracted ID. Examples:")
        for pid in inconsistent.index[:5]:
            print(f"  {pid}: {df.loc[df[person_col] == pid, label_col].value_counts().to_dict()}")
        print("Using each person's majority label for splitting purposes.\n")

    return grouped.agg(lambda s: s.value_counts().idxmax())  # index=person_id, value=majority SCC label


def check_no_person_leakage(train_df, val_df, test_df, person_col):
    train_people = set(train_df[person_col])
    val_people = set(val_df[person_col])
    test_people = set(test_df[person_col])

    overlaps = {
        "train/val": train_people & val_people,
        "train/test": train_people & test_people,
        "val/test": val_people & test_people,
    }
    leaks = {pair: ids for pair, ids in overlaps.items() if ids}
    if leaks:
        print("PERSON-LEVEL LEAKAGE CHECK FAILED:", file=sys.stderr)
        for pair, ids in leaks.items():
            print(f"  {pair}: {len(ids)} shared person IDs, e.g. {sorted(ids)[:5]}", file=sys.stderr)
        sys.exit(1)

    total = len(train_people) + len(val_people) + len(test_people)
    print(f"OK: no person appears in more than one split ({total} unique people across all three).")


def generate_person_splits(person_label):
    """70/20/10 stratified split on PEOPLE (not photos), by each person's majority SCC label."""
    person_ids = person_label.index.to_series()

    train_ids, temp_ids = train_test_split(
        person_ids, test_size=0.30, stratify=person_label, random_state=SEED
    )
    temp_labels = person_label.loc[temp_ids]
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=1 / 3, stratify=temp_labels, random_state=SEED
    )
    return set(train_ids), set(val_ids), set(test_ids)


def augment_image(
    image,
    hflip_prob: float = 0.5,
    max_rotation_deg: float = 15.0,
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
):
    """Random horizontal flip + rotation + color jitter on a single PIL image.
    Call this ONLY at training time, on images listed in train.csv -- never
    on val/test, and never before/during split generation."""
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
            sys.exit(f"ERROR: missing split file(s): {[str(p) for p in missing]}")
        train_df, val_df, test_df = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)
        for d in (train_df, val_df, test_df):
            d["person_id"] = extract_person_id(d, args.filename_column)
        check_no_person_leakage(train_df, val_df, test_df, "person_id")
        return

    existing = [p for p in (train_path, val_path, test_path) if p.exists()]
    if existing and not args.force:
        names = ", ".join(p.name for p in existing)
        sys.exit(
            f"ERROR: {names} already exist -- pass --force to regenerate. Required this run: "
            f"the existing files were built by the old per-photo (leaky) split."
        )

    if not args.manifest.exists():
        sys.exit(f"ERROR: manifest not found at {args.manifest.resolve()}")

    df = pd.read_csv(args.manifest)

    for col in (args.filename_column, args.label_column):
        if col not in df.columns:
            sys.exit(f"ERROR: column '{col}' not found. Available columns: {list(df.columns)}")

    if df[args.label_column].isna().any():
        sys.exit(f"ERROR: {int(df[args.label_column].isna().sum())} rows have a null {args.label_column}.")

    df["person_id"] = extract_person_id(df, args.filename_column)
    person_label = build_person_table(df, "person_id", args.label_column)

    train_people, val_people, test_people = generate_person_splits(person_label)

    train_df = df[df["person_id"].isin(train_people)]
    val_df = df[df["person_id"].isin(val_people)]
    test_df = df[df["person_id"].isin(test_people)]

    check_no_person_leakage(train_df, val_df, test_df, "person_id")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    n = len(df)
    print(f"\nWrote frozen, person-grouped splits to {out_dir.resolve()}:")
    print(f"  train.csv: {len(train_df):5d} images, {len(train_people):5d} people ({len(train_df)/n:.1%} of images)")
    print(f"  val.csv:   {len(val_df):5d} images, {len(val_people):5d} people ({len(val_df)/n:.1%} of images)")
    print(f"  test.csv:  {len(test_df):5d} images, {len(test_people):5d} people ({len(test_df)/n:.1%} of images)")

    print("\nSCC_label distribution per split (should look roughly similar across all three):")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name}: {d[args.label_column].value_counts(normalize=True).sort_index().round(3).to_dict()}")

    print("\nReminder: augment_image() must only be applied when loading images from")
    print("train.csv, at training time -- never to val/test, never before this split.")


if __name__ == "__main__":
    main()