# Throwaway diagnostic -- not part of the pipeline, safe to delete after use.
# Run from the HueView-tool repo root, same as before.

from pathlib import Path
import pandas as pd

IMAGES_ROOT = Path("data/processed/images")
CANDIDATES = [
    "data/processed/resolved_manifest.csv",
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/test.csv",
]

disk_files = [p for p in IMAGES_ROOT.rglob("*") if p.is_file()]
disk_names = {p.name for p in disk_files}
print(f"on-disk files: {len(disk_files)} total, {len(disk_names)} unique names\n")

all_filenames = {}

for path_str in CANDIDATES:
    path = Path(path_str)
    if not path.exists():
        print(f"=== {path_str} === MISSING\n")
        continue

    df = pd.read_csv(path)
    print(f"=== {path_str} ===")
    print("columns:", list(df.columns))
    print("rows:", len(df))

    fname_col = None
    for c in df.columns:
        if c.lower() in ("filename", "file_name", "file", "image", "image_name"):
            fname_col = c
            break

    if fname_col is None:
        print("no obvious filename column found -- skipping match check\n")
        continue

    print(f"using column '{fname_col}' as filename")
    sample = df[fname_col].astype(str).head(3).tolist()
    print("sample values (repr):", [repr(s) for s in sample])

    names = set(df[fname_col].astype(str))
    raw_matches = len(names & disk_names)
    basenames = {Path(n).name for n in names}
    basename_matches = len(basenames & disk_names)

    print(f"exact matches against disk: {raw_matches} / {len(names)}")
    print(f"matches after taking just the basename (stripping any folder prefix): {basename_matches} / {len(names)}")
    print()

    all_filenames[path_str] = names

if "data/processed/resolved_manifest.csv" in all_filenames:
    resolved = all_filenames["data/processed/resolved_manifest.csv"]
    splits_union = set()
    for k in ("data/processed/train.csv", "data/processed/val.csv", "data/processed/test.csv"):
        if k in all_filenames:
            splits_union |= all_filenames[k]
    if splits_union:
        print(f"train+val+test unique filenames: {len(splits_union)}")
        print(f"of those, also present in resolved_manifest.csv: {len(splits_union & resolved)}")