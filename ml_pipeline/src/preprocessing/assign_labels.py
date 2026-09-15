"""
HueView — Phase 3, Step 5: Label every image Low/Medium/High.

Uses the boundaries derived from the real K-Means fit (run in Colab, since
scikit-learn is blocked locally on this machine). This step itself only
needs pandas, so it runs fine locally.
"""

import json
import pandas as pd
from pathlib import Path

# From the Colab K-Means run on the full 43,221-image dataset:
CENTROIDS = [82.19, 126.54, 208.77]
B1 = 104.37
B2 = 167.66

df = pd.read_csv("data/processed/step2_luminance.csv")


def assign_label(mean_y):
    if mean_y < B1:
        return "Low"
    elif mean_y <= B2:
        return "Medium"
    else:
        return "High"


df["illumination_label"] = df["mean_y"].apply(assign_label)
df.to_csv("data/processed/manifest.csv", index=False)

# Save the boundaries durably now that we're back on local ground --
# Phase 14 (inference) reuses these exact numbers later, unchanged.
boundaries = {"centroids": CENTROIDS, "b1": B1, "b2": B2}
with open("data/processed/illumination_boundaries.json", "w") as f:
    json.dump(boundaries, f, indent=2)

print("Illumination distribution (sanity check):")
print(df["illumination_label"].value_counts())
print(f"\nSaved {len(df)} rows to data/processed/manifest.csv")
print("Saved boundaries to data/processed/illumination_boundaries.json")