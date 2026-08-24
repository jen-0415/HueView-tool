"""
HueView — Phase 3, Step 3-4: Fit K-Means on the real brightness distribution
and derive the Low/Medium/High illumination boundaries.

Reads data/processed/step2_luminance.csv (built by extract_luminance.py).
Saves the derived boundaries to illumination_boundaries.json — Phase 14
(inference pipeline) reuses these exact two numbers later; it never refits
K-Means on a single new image.
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

df = pd.read_csv("data/processed/step2_luminance.csv")
y_values = df["mean_y"].to_numpy().reshape(-1, 1)

kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
kmeans.fit(y_values)

centroids = sorted(kmeans.cluster_centers_.flatten().tolist())  # dark -> bright
b1 = (centroids[0] + centroids[1]) / 2   # Low/Medium cutoff
b2 = (centroids[1] + centroids[2]) / 2   # Medium/High cutoff

print(f"Centroids (Low, Medium, High): {[round(c, 2) for c in centroids]}")
print(f"Boundaries: b1={b1:.2f}, b2={b2:.2f}")

# Save durably -- don't let these live only in a terminal scrollback.
boundaries = {"centroids": centroids, "b1": b1, "b2": b2}
output_path = Path("data/processed/illumination_boundaries.json")
with open(output_path, "w") as f:
    json.dump(boundaries, f, indent=2)
print(f"Saved boundaries to {output_path}")