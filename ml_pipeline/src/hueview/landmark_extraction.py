"""
Phase 7.1 -- MediaPipe Landmark Extraction (Stream A)

Runs MediaPipe's face landmarker on the Phase 3 processed (224x224,
un-normalized) face images and stores the landmarks per image.

Output:
    ml_pipeline/data/processed/landmarks.npy          -- float32, shape (N, 468, 3)
    ml_pipeline/data/processed/landmarks_index.csv    -- row_index, filename, rule, root
    ml_pipeline/data/processed/landmark_failures.csv  -- filename, reason
"""

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ------------------------------------------------------------ project root --


def find_project_root(start: Path, marker: str = "ml_pipeline") -> Path:
    """Walk upward from `start` until a directory containing `marker` is found."""
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
        if candidate.name == marker:
            return candidate.parent
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.src.baseline.path_resolver import resolve_image_path  # noqa: E402

# ---------------------------------------------------------------- config --

OUT_DIR = PROJECT_ROOT / "ml_pipeline" / "data" / "processed"
IMAGES_ROOT = OUT_DIR / "images"

SPLIT_PATHS = [
    OUT_DIR / "train.csv",
    OUT_DIR / "val.csv",
    OUT_DIR / "test.csv",
]

LANDMARKS_PATH = OUT_DIR / "landmarks.npy"
INDEX_PATH = OUT_DIR / "landmarks_index.csv"
FAILURES_PATH = OUT_DIR / "landmark_failures.csv"

MODEL_PATH = PROJECT_ROOT / "face_landmarker.task"
if not MODEL_PATH.is_file():
    MODEL_PATH = PROJECT_ROOT / "ml_pipeline" / "face_landmarker.task"

NUM_LANDMARKS = 468  # model returns 478 (468 mesh + 10 iris); keep the first 468


def build_image_lookup_table() -> tuple[dict[str, Path], dict[str, list[Path]], dict[str, Path]]:
    """Indexes IMAGES_ROOT with multi-level fallbacks (exact path, bare filename, stem only)."""
    print(f"Indexing images directory under {IMAGES_ROOT}...")
    exact_map = {}
    name_to_paths = {}
    stem_map = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if IMAGES_ROOT.is_dir():
        for root, _, files in os.walk(IMAGES_ROOT):
            root_path = Path(root)
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in valid_exts:
                    full_path = root_path / file

                    # 1. Exact relative MST path
                    try:
                        rel_mst = full_path.relative_to(IMAGES_ROOT)
                        exact_map[str(rel_mst).replace("\\", "/")] = full_path
                        exact_map[str(rel_mst)] = full_path
                    except ValueError:
                        pass

                    # 2. Bare filename
                    file_lower = file.lower()
                    if file_lower not in name_to_paths:
                        name_to_paths[file_lower] = []
                    name_to_paths[file_lower].append(full_path)

                    # 3. Stem only
                    stem_map[Path(file).stem.lower()] = full_path

    print(f"Indexed {len(exact_map)} relative paths and {len(name_to_paths)} unique filenames.")
    return exact_map, name_to_paths, stem_map


def resolve_local_path(filename: str, exact_map: dict, name_to_paths: dict, stem_map: dict):
    """Multi-stage resolution fallback including OneDrive duplicate copy handling."""
    str_file = str(filename).replace("\\", "/")

    if "images/" in str_file:
        str_file = str_file.split("images/")[-1]

    # Clean copy suffixes like ' (2)', ' (1)'
    str_file_clean = re.sub(r"\s*\(\d+\)", "", str_file)

    bare_name = Path(str_file).name.lower()
    bare_name_clean = Path(str_file_clean).name.lower()

    stem_name = Path(str_file).stem.lower()
    stem_name_clean = Path(str_file_clean).stem.lower()

    # 1. Exact match (original and clean)
    if str_file in exact_map:
        return exact_map[str_file], "exact_mst_match", "images"
    if str_file_clean in exact_map:
        return exact_map[str_file_clean], "exact_mst_clean_match", "images"

    # 2. Bare filename match (original and clean)
    if bare_name in name_to_paths:
        return name_to_paths[bare_name][0], "bare_filename_match", "images"
    if bare_name_clean in name_to_paths:
        return name_to_paths[bare_name_clean][0], "bare_filename_clean_match", "images"

    # 3. Stem match (original and clean)
    if stem_name in stem_map:
        return stem_map[stem_name], "stem_match", "images"
    if stem_name_clean in stem_map:
        return stem_map[stem_name_clean], "stem_clean_match", "images"

    # 4. Fallback: try appending or removing trailing '_1'
    if f"{stem_name_clean}_1" in stem_map:
        return stem_map[f"{stem_name_clean}_1"], "stem_append_1_match", "images"

    return resolve_image_path(filename)


def load_manifest():
    """Concatenate the three frozen Phase 5 splits."""
    existing_splits = [p for p in SPLIT_PATHS if p.is_file()]
    if not existing_splits:
        raise SystemExit(
            f"ERROR: No split CSV files found in {OUT_DIR}. Ensure train.csv, val.csv, and test.csv exist."
        )
    return pd.concat([pd.read_csv(p) for p in existing_splits], ignore_index=True)


def extract_landmarks(landmarker, image_rgb: np.ndarray):
    """image_rgb: (224, 224, 3) uint8, RGB. Returns (468, 3) float32."""
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
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"ERROR: MediaPipe task model not found at {MODEL_PATH}. Download face_landmarker.task first."
        )

    exact_map, name_to_paths, stem_map = build_image_lookup_table()

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
            filename = row["filename"]

            img_path, rule, root = resolve_local_path(filename, exact_map, name_to_paths, stem_map)

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
            "root and that ml_pipeline/data/processed/images/ contains the batch folders."
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


if __name__ == "__main__":
    run()