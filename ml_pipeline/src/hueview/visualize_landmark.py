"""
Visualize MediaPipe face landmarks with their index numbers on one of your
real processed images, so you can pick which indices belong to each of the
5 HueView regions by looking, rather than guessing from memory.

Uses the same Tasks-API FaceLandmarker + face_landmarker.task model file
your 7.1 landmark extraction already uses.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

IMAGE_PATH = "data/processed/faces/MST-1/28_Brazilian Faces28-06_face_1.jpg"  # <-- pick any one real processed image
MODEL_PATH = "face_landmarker.task"  # matches what 7.1 already uses
OUTPUT_PATH = "landmark_reference.png"
UPSCALE = 4  # blow the 224x224 image up so index labels are actually readable

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(options)

mp_image = mp.Image.create_from_file(IMAGE_PATH)
result = landmarker.detect(mp_image)

if not result.face_landmarks:
    raise SystemExit(f"No face detected in {IMAGE_PATH} -- try a different sample image.")

landmarks = result.face_landmarks[0]  # this face's 468 (or 478 with iris refinement) landmarks

# Load with OpenCV too, upscaled so the index labels are actually legible
image = cv2.imread(IMAGE_PATH)
h, w = image.shape[:2]
image = cv2.resize(image, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
h, w = image.shape[:2]

for idx, lm in enumerate(landmarks):
    x, y = int(lm.x * w), int(lm.y * h)
    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    cv2.putText(image, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

cv2.imwrite(OUTPUT_PATH, image)
print(f"Saved {OUTPUT_PATH} -- open it and zoom into each region to read off index numbers.")