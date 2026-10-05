import numpy as np
import pandas as pd
import cv2

landmarks = np.load("data/processed/landmarks.npy")
index = pd.read_csv("data/processed/landmarks_index.csv")

print("shape:", landmarks.shape)
print("dtype:", landmarks.dtype)
print()
print("Row 0, first 5 of its 468 landmarks (x, y, z):")
print(landmarks[0, :5, :])

# Pick one row and draw its 468 points on the actual face image, so you can
# see what's stored, not just read numbers.
row = 0
filename = index.loc[row, "filename"]
print(f"\nRow {row} corresponds to: {filename}")

resolved = pd.read_csv("data/processed/resolved_manifest.csv")
img_path = resolved.loc[resolved["filename"] == filename, "resolved_path"].iloc[0]

img = cv2.imread(img_path)
points = landmarks[row]  # (468, 3) -- x, y, z for this one image

for x, y, z in points:
    cv2.circle(img, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)

cv2.imwrite("landmark_preview.png", img)
print("Saved landmark_preview.png -- open it to see the 468 points on the face.")