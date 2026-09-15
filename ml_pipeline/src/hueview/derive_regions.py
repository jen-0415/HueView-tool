"""
Derives candidate landmark groups for HueView's 5 regions (forehead, left
cheek, right cheek, nose bridge, jawline), using MediaPipe's OFFICIAL
landmark groupings (face oval, eyebrows, nose) as anchors, plus simple
geometry -- instead of hand-picking all ~30-40 index numbers by eye.

Forehead, jawline, and nose bridge are solidly anchored (derived directly
from official groups + straightforward position rules). Left/right cheek
are the least directly anchored -- MediaPipe doesn't define a "cheek"
group at all, so these are approximated from face-oval + eye position.
Look hardest at the cheek regions when you check the output image.

Draws all 5 candidate regions in different colors on one real photo so you
can visually confirm them at a glance, rather than reading off 30-40
individual numbers cold.

--------------------------------------------------------------------------
v4 -- CHEEK FIX. v3's cheeks were wrong in two independent ways, both
visible in region_reference.png as two wide wedges spanning the whole face:

  1. SIDE MISMATCH. left_band was selected with xs[i] < face_center_x
     (image-LEFT), but left_eye_lower_pts came from MediaPipe's LEFT_EYE,
     which is the SUBJECT's left eye = image-RIGHT. The convex hull
     therefore stretched from the image-left face edge across to the
     image-right eye. Same error mirrored on the other side, so both cheeks
     covered most of the face and overlapped each other almost entirely.

     MediaPipe names its groups by subject anatomy. Table 3 names regions by
     image position ("left 15-45% width"). The two conventions are mirror
     images, so an image-left region must draw from MediaPipe's RIGHT_*
     groups. That is the whole bug.

  2. NOSE BLEED. left_nose_edge / right_nose_edge pulled midline NOSE points
     into the cheeks, so left_cheek shared 8 landmark indices with
     nose_bridge (2, 5, 6, 19, 45, 94, 195, 197) and right_cheek shared 2
     (4, 275). The hulls overlapped as a result.

     Regions have to be disjoint. encode_label_map stores one region id per
     pixel, so a shared pixel is silently assigned to whichever region comes
     later in REGION_ORDER and lost from the earlier one; and Phase 8.2's
     15-element CIELAB vector would average shared pixels into two different
     regional means, correlating features the study reports as independent.

Fix: pair each cheek with the eye on the SAME image side, drop the nose
points entirely (Table 3 describes the cheek as "lateral"), and require
cheek points to clear the nose by a lateral margin. A per-side x filter is
also applied to the eye points, so a future edit that swaps the groups back
cannot silently reintroduce bug 1.

An index-level disjointness check now runs at the end and prints a warning
rather than failing silently.
--------------------------------------------------------------------------
"""

from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Reference face the index groups are derived from. Must be a PROCESSED
# 224x224 crop (that is what 7.1 runs landmarks on), and should be frontal,
# neutral, eyes open, no glasses, hair off the forehead. Pose matters here:
# the groups derived from this one face are frozen and applied to every
# image in the dataset, so an unusual pose skews all 43,000 masks.
IMAGE_PATH = "data/processed/images/c1_processed/MST-9/0736_0_0_0_01.jpg"
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

# Aliases in IMAGE space, which is the convention Table 3 uses. Read every
# region rule below in these terms and the mirror confusion disappears.
IMAGE_LEFT_EYE = RIGHT_EYE    # MediaPipe RIGHT_* = subject's right = image-LEFT
IMAGE_RIGHT_EYE = LEFT_EYE    # MediaPipe LEFT_*  = subject's left  = image-RIGHT

# Second mesh ring: every landmark directly connected to a FACE_OVAL point in
# MediaPipe's own tessellation, minus the oval itself. 36 points, one step
# inside the face boundary, and verified disjoint from the eye, brow, lip and
# nose groups.
#
# This exists because FACE_OVAL points ARE the face boundary, so any convex
# hull built from them reaches the ear and sideburn. Ear skin passes an HSV
# skin filter perfectly well, so Phase 7.4 would not catch the contamination —
# it would quietly average ear into the cheek's CIELAB mean. Using the inner
# ring as the lateral boundary keeps the hull on the cheek, and derives that
# boundary from official topology rather than hand-picked indices.
_ADJACENCY = defaultdict(set)
for _edge in flc.FACE_LANDMARKS_TESSELATION:
    _ADJACENCY[_edge.start].add(_edge.end)
    _ADJACENCY[_edge.end].add(_edge.start)
INNER_RING = sorted({n for o in FACE_OVAL for n in _ADJACENCY[o]} - set(FACE_OVAL))

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

    # --- Cheeks (v5) --------------------------------------------------
    # v4 fixed the side mismatch but left the cheeks as thin triangles that
    # reached the ear and missed the apple of the cheek. Two causes:
    #
    #   * dropping the NOSE points (to fix the nose_bridge overlap) left the
    #     cheeks with no medial boundary, so the hull collapsed onto its
    #     lateral points and a diagonal sliced off the cheek's centre;
    #   * FACE_OVAL points are the face boundary, so the hull ran out to the
    #     sideburn and included ear skin.
    #
    # v5 bounds the region on all four sides with real point groups:
    #
    #   top     lower eyelid, same image side
    #   medial  nose wing -- NOSE points not claimed by nose_bridge, so
    #           disjointness holds by construction. Anatomically the wing IS
    #           the medial edge of the cheek. Below the wing the hull closes
    #           on a diagonal, which is correct: that line follows the
    #           nasolabial fold, and the cheek does not extend medially past
    #           it.
    #   lateral INNER_RING instead of FACE_OVAL -- one mesh step in from the
    #           face edge, keeping the hull off the ear.
    #   bottom  min(lips_top_y, jaw_cutoff_y)
    #
    # Result is a quadrilateral over real cheek skin, much closer to Table 3
    # ("middle 30-65% height, left 15-45% width") than v4's triangle.
    lips_top_y = min(ys[i] for i in LIPS)

    # Stop the cheek band at whichever comes first, the top of the lips or the
    # jaw cutoff. Without the min(), a face whose lips sit below the jaw
    # cutoff (common -- jaw_cutoff_y is nose_tip plus a fixed 0.12) gives the
    # cheek and jawline bands an overlapping y-window.
    cheek_bottom_y = min(lips_top_y, jaw_cutoff_y)
    nose_bridge_set = set(nose_bridge)

    def cheek_indices(eye_group, side):
        """side: 'left' or 'right', meaning IMAGE-left / IMAGE-right."""
        eye_mid_y = np.percentile([ys[i] for i in eye_group], 50)
        eye_lower = [i for i in eye_group if ys[i] >= eye_mid_y]
        top_y = min(ys[i] for i in eye_lower)

        if side == "left":
            def correct_side(i):
                return xs[i] < face_center_x
        else:
            def correct_side(i):
                return xs[i] > face_center_x

        def in_band(i):
            return top_y <= ys[i] <= cheek_bottom_y

        lateral = [i for i in INNER_RING if in_band(i) and correct_side(i)]
        wing = [i for i in NOSE
                if i not in nose_bridge_set and in_band(i) and correct_side(i)]
        # Belt-and-braces: even if the eye groups get swapped by a later edit,
        # wrong-side points are dropped here rather than silently widening the
        # hull across the face.
        eye_lower = [i for i in eye_lower if correct_side(i)]

        if not wing:
            print(f"  [warn] {side} cheek has no nose-wing anchor; its medial "
                  f"edge will collapse. Check the reference face's pose.")
        if len(lateral) < 2:
            print(f"  [warn] {side} cheek has only {len(lateral)} lateral "
                  f"point(s); the hull may be degenerate.")

        return sorted(set(lateral + wing + eye_lower))

    left_cheek = cheek_indices(IMAGE_LEFT_EYE, "left")
    right_cheek = cheek_indices(IMAGE_RIGHT_EYE, "right")

    return {
        "forehead": sorted(set(forehead)),
        "left_cheek": sorted(set(left_cheek)),
        "right_cheek": sorted(set(right_cheek)),
        "nose_bridge": sorted(set(nose_bridge)),
        "jawline": sorted(set(jawline)),
    }


def check_disjoint(regions):
    """
    Index-level disjointness check.

    Shared indices guarantee overlapping hulls, so this is the cheap early
    warning. It is necessary but not sufficient -- two regions with no shared
    indices can still produce overlapping convex hulls, which is what
    `run_region_qa.py --overlap-only` measures in pixels.
    """
    from itertools import combinations
    clashes = []
    for a, b in combinations(regions, 2):
        shared = sorted(set(regions[a]) & set(regions[b]))
        if shared:
            clashes.append((a, b, shared))
    if clashes:
        print("\n!! REGIONS SHARE LANDMARK INDICES -- hulls will overlap:")
        for a, b, shared in clashes:
            print(f"     {a} n {b}: {shared}")
        print("   Fix before pasting these groups into segment_regions.py.")
    else:
        print("\nDisjointness (index level): OK, no shared landmarks.")
    return not clashes


if __name__ == "__main__":
    from pathlib import Path

    # Check paths BEFORE loading the model. MediaPipe + TensorFlow import
    # takes the better part of a minute on Windows, and a missing file
    # otherwise surfaces as a FileNotFoundError at the very end of that.
    if not Path(IMAGE_PATH).exists():
        print(f"IMAGE_PATH does not exist: {IMAGE_PATH}")
        print("Run from the repo root (paths here are relative), and set")
        print("IMAGE_PATH near the top of this file to a real processed crop.")
        root = Path("data/processed/images")
        if root.exists():
            print("\nSome candidates:")
            for p in list(root.rglob("*.jpg"))[:5]:
                print(f"  {p.as_posix()}")
        raise SystemExit(1)

    if not Path(MODEL_PATH).exists():
        raise SystemExit(f"MODEL_PATH does not exist: {MODEL_PATH} (expected at the repo root)")

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

    check_disjoint(regions)

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
    print(f"\nSaved {OUTPUT_PATH} -- green (left_cheek) should now sit on the IMAGE-LEFT")
    print("cheek only and orange (right_cheek) on the IMAGE-RIGHT cheek only, with a")
    print("visible gap between them across the nose. If either still crosses the")
    print("midline, stop and re-check before pasting indices into segment_regions.py.")
