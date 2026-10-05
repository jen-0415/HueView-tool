# Throwaway diagnostic -- not part of the pipeline, safe to delete after use.
# Run from the HueView-tool repo root. This may take a few minutes -- it
# opens every fallback-recovered image, same as resolve_manifest.py does.

from pathlib import Path

import cv2
import pandas as pd

TOL = 0.5


def mean_y(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].mean())


splits = pd.concat([
    pd.read_csv("data/processed/train.csv"),
    pd.read_csv("data/processed/val.csv"),
    pd.read_csv("data/processed/test.csv"),
], ignore_index=True)[["filename", "mean_y"]]

corrections = pd.read_csv("data/processed/path_corrections.csv")
df = corrections.merge(splits, on="filename", how="left")

print(f"Checking {len(df)} fallback-recovered rows against their stored mean_Y...\n")

verified = []
for _, row in df.iterrows():
    stored = row["mean_y"]
    actual = mean_y(row["actual_path_used"])
    ok = actual is not None and pd.notna(stored) and abs(actual - stored) <= TOL
    verified.append(ok)

df["verified"] = verified
n_ok = sum(verified)
n_bad = len(df) - n_ok

print(f"Confirmed correct (mean_Y matches within {TOL}): {n_ok} / {len(df)}")
print(f"MISMATCHED (wrong file likely used):            {n_bad} / {len(df)}")

if n_bad:
    bad = df[~df["verified"]].copy()
    bad.to_csv("data/processed/fallback_mismatches.csv", index=False)
    print(f"\nWrote {n_bad} mismatched rows to fallback_mismatches.csv")
    print("\nSample:")
    print(bad[["filename", "recorded_root", "actual_path_used", "mean_y"]].head(10).to_string())
    print("\nIf this count is non-trivial, the next step is searching the other")
    print("batch folders for a mean_Y match on these specific rows -- the same")
    print("verification resolve_manifest.py already does, just scoped to this subset.")