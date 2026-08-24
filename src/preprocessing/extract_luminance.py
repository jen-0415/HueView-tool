"""
HueView — Phase 3, Step 2: Extract each image's brightness (mean Y / luminance)

Walks data/faces/MST-1 ... MST-10, computes the mean Y (YCrCb luminance) for
every valid image, and saves the result to a CSV. That CSV is the input to
Step 3 (the actual K-Means fit in cluster_illumination.py).
"""

import cv2
import pandas as pd
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
face_dir = Path("data/processed/faces")


def mean_luminance(face_rgb):
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    return float(ycrcb[:, :, 0].mean())


results = []
skipped_ext = []
skipped_unreadable = []

for img_path in face_dir.rglob("*"):
    if img_path.is_dir():
        continue

    if img_path.suffix.lower() not in VALID_EXTENSIONS:
        skipped_ext.append(str(img_path))
        continue

    mst_label = img_path.parent.name
    relative_path = f"{mst_label}/{img_path.name}"

    face = cv2.imread(str(img_path))
    if face is None:
        skipped_unreadable.append(str(img_path))
        continue

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    results.append({
        "filename": relative_path,
        "mst_label": mst_label,
        "mean_y": mean_luminance(face),
    })

# ---- Summary -----------------------------------------------------------
print(f"Processed {len(results)} images")
print(f"Skipped (wrong extension): {len(skipped_ext)}")
print(f"Skipped (unreadable/corrupt): {len(skipped_unreadable)}")
if skipped_ext:
    print("Example skipped-extension files:", skipped_ext[:10])
if skipped_unreadable:
    print("Example unreadable files:", skipped_unreadable[:10])

# ---- Save so Step 3 doesn't require re-scanning every image ------------
output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(output_path / "step2_luminance.csv", index=False)
print(f"Saved {len(results)} rows to {output_path / 'step2_luminance.csv'}")