"""
Phase 7.1 -- MediaPipe Landmark Extraction (Stream A)

Runs MediaPipe's face landmarker on the Phase 3 processed (224x224,
un-normalized) face images and stores the landmarks per image. Phase 7.3
uses these to build the 5 regional boundaries -- they must come from the
ORIGINAL image, not the SSR output from 7.2 (SSR strips the contrast the
landmarker needs to place points accurately).

NOTE ON THE MEDIAPIPE VERSION: the classic `mp.solutions.face_mesh.FaceMesh`
API used in most tutorials (and matching the "468 landmarks" framing in the
plan) has been removed from current mediapipe releases -- `pip install
mediapipe` today only ships the newer Tasks API
(`mediapipe.tasks.vision.FaceLandmarker`), confirmed against mediapipe
0.10.33. Its default model returns 478 points: the classic 468-point mesh
plus 10 iris points appended at indices 468-477. This script keeps just the
first 468 so your downstream phases match the plan's numbers exactly -- see
`coords[:NUM_LANDMARKS]` below.

PATH RESOLUTION -- FIXED. This script now calls
baseline.path_resolver.resolve_image_path() instead of doing its own
folder search. The previous in-script search had two defects that
resolve_image_path() already handles correctly, and that every other
script in this pipeline already relies on it for:

  1. SUFFIX ROUTING (the serious one). The old code stripped " (2)" from
     the filename and then searched folders in the order processed,
     c1_processed, c2_processed, v5_processed, taking the first hit. But
     a "(2)" row's SCC and illumination labels were computed from the
     c2_processed rendering of that image -- so matching processed/ first
     pairs the row's labels with a DIFFERENT image. path_resolver.py's
     own docstring flags this explicitly: "Stripping the suffix and
     serving the `processed` version pairs ~28% of the dataset with
     labels derived from a different image. Nothing errors -- the numbers
     just come out wrong." SUFFIX_ROUTING maps suffix 2 -> c2_processed
     first, which is what the labels actually describe.

  2. V5 EXTENSION MISMATCH. The v5 batch is recorded in the manifest with
     a .png extension but stored on disk as .bmp. The old code built the
     path with the manifest's extension and never tried .bmp, so all
     2,497 v5 images were logged as "file_not_found" despite being
     present locally the whole time. resolve_image_path() has an explicit
     v5_bmp rule for this.

The `path_source` column in landmarks_index.csv now records which rule
matched (exact / suffix_routed / v5_bmp / fallback) and which batch root
the image actually came from, so the resolution is auditable after the
fact rather than invisible.

NOTE ON RUNNING DIRECTORY: path_resolver.py anchors its roots on the
relative path "data/processed/images", so this script must be run from
the repo root (as all the other pipeline scripts are). Running it from
elsewhere will resolve nothing.

Input:
    data/processed/train.csv, val.csv, test.csv -- Phase 5 frozen splits:
                                          filename, mst_label, mean_y,
                                          illumination_label, SCC_label,
                                          person_id
    data/processed/images/<batch>/<MST-N>/<file> -- resolved via
                                          path_resolver.resolve_image_path

Output:
    data/processed/landmarks.npy         -- float32, shape (N, 468, 3)
    data/processed/landmarks_index.csv    -- row_index, filename, rule, root
    data/processed/landmark_failures.csv  -- filename, reason
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm


def find_project_root(start: Path, marker: str = "data") -> Path:
    """Walk upward from `start` until a directory containing `marker` is found."""
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline.path_resolver import resolve_image_path  # noqa: E402

SPLIT_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "train.csv",
    PROJECT_ROOT / "data" / "processed" / "val.csv",
    PROJECT_ROOT / "data" / "processed" / "test.csv",
]
OUT_DIR = PROJECT_ROOT / "data" / "processed"

LANDMARKS_PATH = OUT_DIR / "landmarks.npy"
INDEX_PATH = OUT_DIR / "landmarks_index.csv"
FAILURES_PATH = OUT_DIR / "landmark_failures.csv"

MODEL_PATH = PROJECT_ROOT / "face_landmarker.task"
NUM_LANDMARKS = 468  # model returns 478 (468 mesh + 10 iris); keep the first 468


def load_manifest():
    """Concatenate the three frozen Phase 5 splits."""
    return pd.concat([pd.read_csv(p) for p in SPLIT_PATHS], ignore_index=True)


def extract_landmarks(landmarker, image_rgb: np.ndarray):
    """image_rgb: (224, 224, 3) uint8, RGB. Returns (468, 3) float32
    [x_px, y_px, z_rel], or None if no face was found."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]  # num_faces=1
    h, w = image_rgb.shape[:2]
    coords = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face], dtype=np.float32)
    return coords[:NUM_LANDMARKS]


def run():
    print(f"Project root: {PROJECT_ROOT}")
    manifest = load_manifest()
    print(f"Total images to process: {len(manifest)}")

    landmarks_list = []
    index_rows = []
    failures = []

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Phase 7.1"):
            filename = row["filename"]  # e.g. "MST-3/some_image.jpg"

            # Suffix-aware, multi-root, extension-aware resolution -- see
            # the PATH RESOLUTION note in the module docstring.
            img_path, rule, root = resolve_image_path(filename)

            if img_path is None:
                failures.append({"filename": filename, "reason": "file_not_found"})
                continue

            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                failures.append({"filename": filename, "reason": "file_unreadable"})
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            coords = extract_landmarks(landmarker, image_rgb)

            if coords is None:
                failures.append({"filename": filename, "reason": "no_face_landmarks"})
                continue

            index_rows.append({
                "row_index": len(landmarks_list),
                "filename": filename,
                "rule": rule,
                "root": root,
            })
            landmarks_list.append(coords)

    if not landmarks_list:
        raise SystemExit(
            "No landmarks extracted at all -- confirm you are running from the repo "
            "root (path_resolver.py uses relative paths) and that "
            "data/processed/images/ contains the batch folders."
        )

    landmarks_array = np.stack(landmarks_list, axis=0)  # (N, 468, 3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(LANDMARKS_PATH, landmarks_array)
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(INDEX_PATH, index=False)
    pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)

    print(f"\nLandmarks extracted: {len(landmarks_list)} / {len(manifest)} images")
    print(f"Failed (logged to {FAILURES_PATH.name}): {len(failures)}")
    if failures:
        print(pd.DataFrame(failures)["reason"].value_counts().to_string())
    print(f"Saved {LANDMARKS_PATH} with shape {landmarks_array.shape}")

    print("\nRESOLUTION RULE USAGE (how each image's real file was located)")
    print("-" * 62)
    print(index_df["rule"].value_counts().to_string())
    print("\nBATCH ROOT USAGE")
    print("-" * 62)
    print(index_df["root"].value_counts().to_string())


if __name__ == "__main__":
    run()
