# Throwaway diagnostic -- not part of the pipeline, safe to delete after use.
# Run from ml_pipeline/:  python scripts/diagnostics/check_alternate_roots.py

from pathlib import Path
import pandas as pd

IMAGES_ROOT = Path("data/processed/images")
SPLIT_PATHS = [
    Path("data/processed/train.csv"),
    Path("data/processed/val.csv"),
    Path("data/processed/test.csv"),
]
RESOLVED_MANIFEST_PATH = Path("data/processed/resolved_manifest.csv")
FAILURES_PATH = Path("data/processed/landmark_failures.csv")


def suffix_after_root(resolved_path):
    """Everything after .../images/<root>/ -- e.g. 'MST-10/0008_1_0_0_01.jpg'."""
    parts = Path(resolved_path).parts
    idx = parts.index("images") + 2  # skip 'images' and the root segment itself
    return Path(*parts[idx:])


manifest = pd.concat([pd.read_csv(p) for p in SPLIT_PATHS], ignore_index=True)
resolved = pd.read_csv(RESOLVED_MANIFEST_PATH)[["filename", "resolved_path", "root", "how"]]
manifest = manifest.merge(resolved, on="filename", how="left")

failures = pd.read_csv(FAILURES_PATH)
unreadable = failures[failures["reason"] == "file_unreadable_or_missing"].merge(
    manifest[["filename", "resolved_path", "root", "how"]], on="filename", how="left"
)
print(f"file_unreadable_or_missing rows: {len(unreadable)}")

print("\nfailures by 'how':")
print(unreadable["how"].value_counts(dropna=False))
print("\nfailures by 'root' (as recorded in resolved_manifest.csv):")
print(unreadable["root"].value_counts(dropna=False))

batch_folders = sorted(p.name for p in IMAGES_ROOT.iterdir() if p.is_dir())
print(f"\nbatch folders present on disk: {batch_folders}")

recovered_by_root = {b: 0 for b in batch_folders}
still_missing = 0

for _, row in unreadable.iterrows():
    suffix = suffix_after_root(row["resolved_path"])
    found = False
    for b in batch_folders:
        if (IMAGES_ROOT / b / suffix).exists():
            recovered_by_root[b] += 1
            found = True
            break
    if not found:
        still_missing += 1

recovered_total = len(unreadable) - still_missing
print(f"\nof {len(unreadable)} failures, found under a DIFFERENT batch folder: {recovered_total}")
print("broken down by which folder actually had it:")
for b, c in recovered_by_root.items():
    print(f"  {b}: {c}")
print(f"still not found anywhere: {still_missing}")