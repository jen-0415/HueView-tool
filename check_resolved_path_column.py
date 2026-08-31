# Throwaway diagnostic -- not part of the pipeline, safe to delete after use.
# Run from the HueView-tool repo root, same as before.

from pathlib import Path
import pandas as pd

IMAGES_ROOT = Path("data/processed/images")
RESOLVED_PATH = Path("data/processed/resolved_manifest.csv")


def rel(p, base):
    return str(p.relative_to(base)).replace("\\", "/")


disk_files = [p for p in IMAGES_ROOT.rglob("*") if p.is_file()]
disk_names = {p.name for p in disk_files}
disk_rel_to_images = {rel(p, IMAGES_ROOT) for p in disk_files}
disk_rel_to_processed = {rel(p, Path("data/processed")) for p in disk_files}

df = pd.read_csv(RESOLVED_PATH)
print("columns:", list(df.columns))
print("rows:", len(df))

print("\n--- 'how' value counts ---")
print(df["how"].value_counts(dropna=False))

print("\n--- 'root' value counts (top 20) ---")
print(df["root"].value_counts(dropna=False).head(20))

print("\n--- first 8 full rows ---")
with pd.option_context("display.max_colwidth", 80, "display.width", 200):
    print(df.head(8).to_string())

resolved_path_norm = df["resolved_path"].astype(str).str.replace("\\", "/", regex=False)
root_norm = df["root"].astype(str).str.replace("\\", "/", regex=False)

print("\n--- path-join candidates vs on-disk files ---")
print("resolved_path as data/processed/<resolved_path>:",
      resolved_path_norm.isin(disk_rel_to_processed).sum(), "/", len(df))
print("resolved_path as images/<resolved_path>:",
      resolved_path_norm.isin(disk_rel_to_images).sum(), "/", len(df))
print("images/<root>/<resolved_path>:",
      (root_norm + "/" + resolved_path_norm).isin(disk_rel_to_images).sum(), "/", len(df))
print("basename(resolved_path) vs any on-disk filename:",
      resolved_path_norm.apply(lambda x: Path(x).name).isin(disk_names).sum(), "/", len(df))