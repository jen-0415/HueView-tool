# Throwaway diagnostic -- not part of the pipeline, safe to delete after use.
# Run from the HueView-tool repo root, same as before.

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SPLIT_PATHS = [
    Path("data/processed/train.csv"),
    Path("data/processed/val.csv"),
    Path("data/processed/test.csv"),
]
RESOLVED_MANIFEST_PATH = Path("data/processed/resolved_manifest.csv")
FAILURES_PATH = Path("data/processed/landmark_failures.csv")


def has_non_ascii(s):
    try:
        str(s).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def imread_unicode(path):
    """Standard workaround for cv2.imread() failing on non-ASCII Windows paths."""
    try:
        stream = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_COLOR)
    except Exception:
        return None


manifest = pd.concat([pd.read_csv(p) for p in SPLIT_PATHS], ignore_index=True)
resolved = pd.read_csv(RESOLVED_MANIFEST_PATH)[["filename", "resolved_path", "root", "how"]]
manifest = manifest.merge(resolved, on="filename", how="left")

failures = pd.read_csv(FAILURES_PATH)
unreadable = failures[failures["reason"] == "file_unreadable_or_missing"].merge(
    manifest[["filename", "resolved_path"]], on="filename", how="left"
)
print(f"file_unreadable_or_missing rows: {len(unreadable)}")

exists_mask = unreadable["resolved_path"].apply(lambda p: Path(p).exists() if pd.notna(p) else False)
print(f"of those, the path actually exists on disk: {exists_mask.sum()} / {len(unreadable)}")

non_ascii_mask = unreadable["resolved_path"].apply(lambda p: has_non_ascii(p) if pd.notna(p) else False)
print(f"of those, the path contains non-ASCII characters: {non_ascii_mask.sum()} / {len(unreadable)}")

print("\nsample failing paths (first 10, repr):")
for p in unreadable["resolved_path"].head(10):
    print(" ", repr(p))

existing_unreadable = unreadable[exists_mask]
sample = existing_unreadable["resolved_path"].head(30)
print(f"\ntesting cv2.imread() vs the Unicode-safe workaround on "
      f"{len(sample)} of the existing-but-failed files:")

imread_fail = 0
workaround_success = 0
for p in sample:
    a = cv2.imread(str(p))
    b = imread_unicode(p)
    if a is None:
        imread_fail += 1
    if b is not None:
        workaround_success += 1

print(f"cv2.imread() failed: {imread_fail} / {len(sample)}")
print(f"np.fromfile + cv2.imdecode succeeded: {workaround_success} / {len(sample)}")