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

NOTE ON THE MANIFEST: this reads the frozen Phase 5 split (train.csv +
val.csv + test.csv), not manifest.csv (a stale pilot-run artifact that
doesn't match any file on disk). Each split row's `filename` is a logical
name from before Phase 3's output got spread across four processing
batches (c1_processed, c2_processed, processed, v5_processed);
resolved_manifest.csv maps each one to a `resolved_path`, the `root` batch
it's supposedly in, and `how` confidently it was matched.

NOTE ON resolved_manifest.csv's ACCURACY: its recorded root/resolved_path
is wrong for a real chunk of rows -- confirmed 13,759 / 43,221 pointed at
root="processed" when the file actually lives elsewhere (13,735 in
c1_processed, 24 in c2_processed). This was 100% of "verified_unique" and
"no_match_used_rule" rows, plus a partial share of "verified_tie" and
"single_candidate". All 13,759 were recoverable by searching the other
batch folders for the same <MST-N>/<filename> suffix -- none were actually
missing. This script tries the recorded resolved_path first and falls back
to that search when it's wrong, logging which path was actually used.
Worth telling whoever built resolved_manifest.csv -- its root-detection
logic looks like it defaults to "processed" under some bug condition, so
even the rows that check out here might only be right by coincidence.

Input:
    data/processed/train.csv, val.csv, test.csv -- Phase 5 frozen splits:
                                          filename, mst_label, mean_y,
                                          illumination_label, SCC_label,
                                          person_id
    data/processed/resolved_manifest.csv -- filename, resolved_path, root,
                                          how
    data/processed/images/<batch>/<MST-N>/<file> -- used as a fallback
                                          search space when resolved_path
                                          doesn't check out

Output:
    data/processed/landmarks.npy                  -- float32, shape
                                                       (N, 468, 3)
    data/processed/landmarks_index.csv             -- row_index, filename,
                                                       how, path_source
    data/processed/landmark_failures.csv           -- filename, reason
    data/processed/low_confidence_resolutions.csv  -- rows where `how` was
                                                       "no_match_used_rule",
                                                       worth a spot check
    data/processed/path_corrections.csv            -- rows where the
                                                       recorded resolved_path
                                                       was wrong and the
                                                       fallback search found
                                                       the real file
                                                       elsewhere
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Config -- adjust these if your repo layout differs from Phase 0
# ---------------------------------------------------------------------------
IMAGES_ROOT = Path("data/processed/images")
SPLIT_PATHS = [
    Path("data/processed/train.csv"),
    Path("data/processed/val.csv"),
    Path("data/processed/test.csv"),
]
RESOLVED_MANIFEST_PATH = Path("data/processed/resolved_manifest.csv")
OUT_DIR = Path("data/processed")

LANDMARKS_PATH = OUT_DIR / "landmarks.npy"
INDEX_PATH = OUT_DIR / "landmarks_index.csv"
FAILURES_PATH = OUT_DIR / "landmark_failures.csv"
LOW_CONFIDENCE_PATH = OUT_DIR / "low_confidence_resolutions.csv"
PATH_CORRECTIONS_PATH = OUT_DIR / "path_corrections.csv"

MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

NUM_LANDMARKS = 468  # classic mesh size -- see the version note above


def ensure_model():
    """Downloads the face landmarker model bundle once, if not already present."""
    if not MODEL_PATH.exists():
        print("Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def load_manifest():
    """
    Concatenates the three frozen Phase 5 splits and attaches each row's
    recorded path by joining against resolved_manifest.csv on `filename`.
    """
    manifest = pd.concat([pd.read_csv(p) for p in SPLIT_PATHS], ignore_index=True)

    resolved = pd.read_csv(RESOLVED_MANIFEST_PATH)[["filename", "resolved_path", "root", "how"]]
    manifest = manifest.merge(resolved, on="filename", how="left")

    unmatched = manifest["resolved_path"].isna().sum()
    if unmatched:
        print(f"WARNING: {unmatched} split rows had no match in "
              f"resolved_manifest.csv -- these will show up in "
              f"{FAILURES_PATH.name} as 'no_resolved_path'.")

    low_conf = manifest[manifest["how"] == "no_match_used_rule"]
    if len(low_conf):
        low_conf[["filename", "resolved_path", "root", "how"]].to_csv(LOW_CONFIDENCE_PATH, index=False)
        print(f"{len(low_conf)} rows were resolved via a fallback rule rather "
              f"than a verified match -- see {LOW_CONFIDENCE_PATH.name}, "
              f"worth a spot check before fully trusting them.")

    return manifest


def discover_batch_folders():
    return sorted(p.name for p in IMAGES_ROOT.iterdir() if p.is_dir())


def resolve_image_path(resolved_path_str, batch_folders):
    """
    Tries the path resolved_manifest.csv recorded first. If it doesn't
    exist -- confirmed to happen for ~32% of rows, concentrated in
    specific `how` categories, see the module note above -- falls back to
    searching the other batch folders for the same <MST-N>/<filename>
    suffix. Returns (path_or_None, source), where source is
    "resolved_manifest", "fallback:<folder>", or None if not found
    anywhere.
    """
    primary = Path(resolved_path_str)
    if primary.exists():
        return primary, "resolved_manifest"

    try:
        parts = primary.parts
        idx = parts.index("images") + 2  # skip 'images' and the recorded root
        suffix = Path(*parts[idx:])
    except ValueError:
        return None, None

    for folder in batch_folders:
        candidate = IMAGES_ROOT / folder / suffix
        if candidate.exists():
            return candidate, f"fallback:{folder}"

    return None, None


def extract_landmarks(landmarker, image_rgb: np.ndarray):
    """
    image_rgb: (224, 224, 3) uint8, RGB -- the Phase 3 processed crop.
    Returns a (468, 3) float32 array [x_px, y_px, z_rel], or None if no
    face was found.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]  # num_faces=1
    h, w = image_rgb.shape[:2]
    coords = np.array(
        [[lm.x * w, lm.y * h, lm.z] for lm in face],
        dtype=np.float32,
    )
    return coords[:NUM_LANDMARKS]


def run():
    ensure_model()
    manifest = load_manifest()
    batch_folders = discover_batch_folders()
    print(f"Batch folders for fallback search: {batch_folders}")

    landmarks_list = []
    index_rows = []
    failures = []
    corrections = []

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Phase 7.1"):
            filename = row["filename"]
            resolved_path = row["resolved_path"]

            if pd.isna(resolved_path):
                failures.append({"filename": filename, "reason": "no_resolved_path"})
                continue

            img_path, path_source = resolve_image_path(resolved_path, batch_folders)

            if img_path is None:
                failures.append({"filename": filename, "reason": "not_found_in_any_batch_folder"})
                continue

            if path_source != "resolved_manifest":
                corrections.append({
                    "filename": filename,
                    "recorded_root": row["root"],
                    "recorded_resolved_path": resolved_path,
                    "actual_path_used": str(img_path),
                })

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
                "how": row["how"],
                "path_source": path_source,
            })
            landmarks_list.append(coords)

    landmarks_array = np.stack(landmarks_list, axis=0)  # (N, 468, 3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(LANDMARKS_PATH, landmarks_array)
    pd.DataFrame(index_rows).to_csv(INDEX_PATH, index=False)
    pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
    if corrections:
        pd.DataFrame(corrections).to_csv(PATH_CORRECTIONS_PATH, index=False)
        print(f"{len(corrections)} images needed a fallback path (recorded "
              f"resolved_path was wrong) -- see {PATH_CORRECTIONS_PATH.name}.")

    print(f"Landmarks extracted: {len(landmarks_list)} / {len(manifest)} images")
    print(f"Failed (logged to {FAILURES_PATH.name}): {len(failures)}")
    print(f"Saved {LANDMARKS_PATH} with shape {landmarks_array.shape}")


if __name__ == "__main__":
    run()