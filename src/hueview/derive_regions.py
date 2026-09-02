"""
Derives candidate landmark groups for HueView's 5 regions (forehead, left
cheek, right cheek, nose bridge, jawline), using MediaPipe's OFFICIAL
landmark groupings (face oval, eyebrows, nose) as anchors, plus simple
geometry -- instead of hand-picking all ~30-40 index numbers by eye.

Forehead, jawline, and nose bridge are solidly anchored (derived directly
from official groups + straightforward position rules). Left/right cheek
are the least directly anchored -- MediaPipe doesn't define a "cheek"
group at all, so these are approximated from face-oval + nose position.
Look hardest at the cheek regions when you check the output image.

Draws all 5 candidate regions in different colors on one real photo so you
can visually confirm them at a glance, rather than reading off 30-40
individual numbers cold.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

IMAGE_PATH = "data/processed/faces/MST-1/28_Brazilian Faces28-06_face_1.jpg"  # <-- pick one real image
MODEL_PATH = "face_landmarker.task"
OUTPUT_PATH = "region_reference.png"
UPSCALE = 4

# --- Pull MediaPipe's OFFICIAL landmark groups (not invented) ---
flc = mp_vision.FaceLandmarksConnections


def unique_indices(connections):
    idxs = set()
    for c in connections:
        idxs.add(c.start)
        idxs.add(c.end)
    return sorted(idxs)


FACE_OVAL = unique_indices(flc.FACE_LANDMARKS_FACE_OVAL)
LEFT_EYEBROW = unique_indices(flc.FACE_LANDMARKS_LEFT_EYEBROW)   # subject's own left (image-right)
RIGHT_EYEBROW = unique_indices(flc.FACE_LANDMARKS_RIGHT_EYEBROW)  # subject's own right (image-left)
NOSE = unique_indices(flc.FACE_LANDMARKS_NOSE)
LEFT_EYE = unique_indices(flc.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE = unique_indices(flc.FACE_LANDMARKS_RIGHT_EYE)
LIPS = unique_indices(flc.FACE_LANDMARKS_LIPS)

NOSE_TIP_IDX = 4  # commonly-referenced nose-tip landmark, present in the official NOSE set
FACE_CENTER_IDX = 1  # commonly-referenced nose/face vertical-midline landmark


def derive_regions(landmarks):
    """landmarks: list of 468 (x, y) normalized coordinates.
    Returns a dict of region_name -> list of landmark indices."""
    ys = {i: landmarks[i][1] for i in range(len(landmarks))}
    xs = {i: landmarks[i][0] for i in range(len(landmarks))}

    eyebrow_y = np.mean([ys[i] for i in LEFT_EYEBROW + RIGHT_EYEBROW])
    nose_tip_y = ys[NOSE_TIP_IDX]
    jaw_cutoff_y = nose_tip_y + 0.12  # below this = jaw/chin territory
    face_center_x = xs[FACE_CENTER_IDX]

    # --- Forehead: face-oval points above the eyebrow line, plus the
    # eyebrows themselves as the region's lower boundary ---
    forehead = [i for i in FACE_OVAL if ys[i] < eyebrow_y] + LEFT_EYEBROW + RIGHT_EYEBROW

    # --- Jawline: face-oval points below the jaw cutoff ---
    jawline = [i for i in FACE_OVAL if ys[i] > jaw_cutoff_y]

    # --- Nose bridge: nose points close to the vertical midline (excludes
    # the wider nostril/wing points, which sit further from center) ---
    nose_x_spread = [abs(xs[i] - face_center_x) for i in NOSE]
    bridge_threshold = np.percentile(nose_x_spread, 50)  # keep the more central half
    nose_bridge = [i for i in NOSE if abs(xs[i] - face_center_x) <= bridge_threshold]

    # --- Cheeks: approximated, least directly anchored -- MediaPipe has no
    # official "cheek" group. v1 was too tall (eyebrow-to-jaw, included the
    # whole eye -> huge wedge). v2 over-corrected to too thin (eye-to-
    # nose-tip -> a handful of sparse points, degenerate sliver). v3: use
    # the LOWER HALF of the eye as a real point set for the top boundary,
    # and the TOP of the lips (an official group) as the bottom boundary --
    # gives the region actual vertical thickness anchored by real points
    # on both ends, not just a narrow y-threshold. ---
    lips_top_y = min(ys[i] for i in LIPS)

    left_eye_ys = [ys[i] for i in LEFT_EYE]
    left_eye_mid_y = np.percentile(left_eye_ys, 50)
    left_eye_lower_pts = [i for i in LEFT_EYE if ys[i] >= left_eye_mid_y]
    cheek_top_y_left = min(ys[i] for i in left_eye_lower_pts)

    right_eye_ys = [ys[i] for i in RIGHT_EYE]
    right_eye_mid_y = np.percentile(right_eye_ys, 50)
    right_eye_lower_pts = [i for i in RIGHT_EYE if ys[i] >= right_eye_mid_y]
    cheek_top_y_right = min(ys[i] for i in right_eye_lower_pts)

    left_band = [i for i in FACE_OVAL if cheek_top_y_left <= ys[i] <= lips_top_y and xs[i] < face_center_x]
    left_nose_edge = [i for i in NOSE if cheek_top_y_left <= ys[i] <= lips_top_y and xs[i] < face_center_x]
    left_cheek = left_band + left_nose_edge + left_eye_lower_pts

    right_band = [i for i in FACE_OVAL if cheek_top_y_right <= ys[i] <= lips_top_y and xs[i] > face_center_x]
    right_nose_edge = [i for i in NOSE if cheek_top_y_right <= ys[i] <= lips_top_y and xs[i] > face_center_x]
    right_cheek = right_band + right_nose_edge + right_eye_lower_pts

    return {
        "forehead": sorted(set(forehead)),
        "left_cheek": sorted(set(left_cheek)),
        "right_cheek": sorted(set(right_cheek)),
        "nose_bridge": sorted(set(nose_bridge)),
        "jawline": sorted(set(jawline)),
    }


# --- Run detection on a real image ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options, running_mode=mp_vision.RunningMode.IMAGE, num_faces=1,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(options)

mp_image = mp.Image.create_from_file(IMAGE_PATH)
result = landmarker.detect(mp_image)

if not result.face_landmarks:
    raise SystemExit(f"No face detected in {IMAGE_PATH} -- try a different sample image.")

face_landmarks = result.face_landmarks[0]
landmarks = [(lm.x, lm.y) for lm in face_landmarks]

regions = derive_regions(landmarks)

print("Derived region index groups:")
for name, idxs in regions.items():
    print(f"  {name} ({len(idxs)} points): {idxs}")

# --- Draw each region in a different color for visual verification ---
image = cv2.imread(IMAGE_PATH)
h, w = image.shape[:2]
image = cv2.resize(image, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
h, w = image.shape[:2]

colors = {
    "forehead": (255, 0, 0),      # blue
    "left_cheek": (0, 255, 0),    # green
    "right_cheek": (0, 165, 255), # orange
    "nose_bridge": (0, 0, 255),   # red
    "jawline": (255, 0, 255),     # magenta
}

for name, idxs in regions.items():
    color = colors[name]
    for i in idxs:
        x, y = int(landmarks[i][0] * w), int(landmarks[i][1] * h)
        cv2.circle(image, (x, y), 4, color, -1)
    # Draw the convex hull outline too, so the region SHAPE is visible, not just dots
    pts = np.array([[int(landmarks[i][0] * w), int(landmarks[i][1] * h)] for i in idxs])
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.polylines(image, [hull], isClosed=True, color=color, thickness=2)

cv2.imwrite(OUTPUT_PATH, image)
print(f"\nSaved {OUTPUT_PATH} -- open it and check each colored region actually covers")
print("the right anatomical area. Look hardest at left_cheek (green) and right_cheek")
print("(orange) -- those are the least directly anchored by official MediaPipe groups.")