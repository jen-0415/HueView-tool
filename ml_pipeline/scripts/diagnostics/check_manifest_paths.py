# Throwaway diagnostic -- not part of the pipeline, safe to delete once
# the filename mismatch is found. Uses the same relative paths as
# landmark_extraction.py, so it needs to be run from the same place.

from pathlib import Path
import pandas as pd

MANIFEST_PATH = Path("data/processed/manifest.csv")
IMAGES_ROOT = Path("data/processed/images")

manifest = pd.read_csv(MANIFEST_PATH)
print("manifest columns:", list(manifest.columns))
print("filename column dtype:", manifest["filename"].dtype)
print("\nfirst 5 manifest 'filename' values (repr):")
for f in manifest["filename"].head(5):
    print(" ", repr(f))

disk_files = [p for p in IMAGES_ROOT.rglob("*") if p.is_file()]
print(f"\ntotal files found on disk: {len(disk_files)}")
print("first 5 on-disk filenames (repr):")
for p in disk_files[:5]:
    print(" ", repr(p.name), "| full path:", p)

manifest_names = set(manifest["filename"].astype(str))
disk_names = {p.name for p in disk_files}
print(f"\nmanifest rows: {len(manifest)}")
print(f"unique on-disk filenames: {len(disk_names)}")
print(f"exact string matches: {len(manifest_names & disk_names)}")