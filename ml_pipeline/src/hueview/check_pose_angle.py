"""
Diagnostic: measure each image's head pose (yaw angle -- left/right turn)
using MediaPipe's facial transformation matrix, so we can see how many
images are angled too far from frontal before deciding on an exclusion
threshold for Phase 7.3's regional segmentation.

This does NOT modify anything -- just measures and reports.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "face_landmarker.task"
FACES_DIR = Path("data/processed/faces")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
    output_facial_transformation_matrixes=True,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(options)


def yaw_from_matrix(matrix):
    """Left/right turn in degrees, extracted from the 4x4 transformation
    matrix's rotation component. 0 = straight-on, positive/negative = turned."""
    r = np.array(matrix)[:3, :3]
    return float(np.degrees(np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))))


results = []
total_files = sum(1 for p in FACES_DIR.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)
processed = 0

for img_path in FACES_DIR.rglob("*"):
    if img_path.suffix.lower() not in VALID_EXTENSIONS:
        continue

    mst_label = img_path.parent.name
    try:
        mp_image = mp.Image.create_from_file(str(img_path))
        result = landmarker.detect(mp_image)
    except Exception as e:
        results.append({"filename": img_path.name, "mst_label": mst_label, "yaw_deg": None, "note": f"error: {e}"})
    else:
        if not result.facial_transformation_matrixes:
            results.append({"filename": img_path.name, "mst_label": mst_label, "yaw_deg": None, "note": "no face/matrix detected"})
        else:
            yaw = yaw_from_matrix(result.facial_transformation_matrixes[0])
            results.append({"filename": img_path.name, "mst_label": mst_label, "yaw_deg": yaw, "note": ""})

    processed += 1
    if processed % 1000 == 0 or processed == total_files:
        print(f"  {processed}/{total_files} images processed...")

df = pd.DataFrame(results)
df.to_csv("data/processed/pose_yaw_diagnostic.csv", index=False)

print(f"Total images checked: {len(df)}")
print(f"Failed (no face/matrix detected): {df['yaw_deg'].isna().sum()}")
print()
print("Yaw angle distribution (degrees, 0 = straight-on):")
print(df["yaw_deg"].describe())
print()
print("How many images fall outside various frontal thresholds:")
for threshold in [15, 30, 45, 60]:
    count = (df["yaw_deg"].abs() > threshold).sum()
    pct = count / len(df) * 100
    print(f"  |yaw| > {threshold}°: {count} images ({pct:.1f}%)")

print(f"\nSaved full per-image results to data/processed/pose_yaw_diagnostic.csv")