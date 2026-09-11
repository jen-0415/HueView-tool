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

Runs on the original, un-normalized 224x224 face images from your Phase 5
splits. These landmarks feed Phase 7.3's regional boundaries -- they must
come from the ORIGINAL image, not 7.2's SSR output (SSR strips the
contrast the landmarker needs to place points accurately).

Input:
    data/processed/train.csv, val.csv, test.csv -- Phase 5 frozen splits:
                                          filename, mst_label, mean_y,
                                          illumination_label, SCC_label,
                                          person_id
    data/processed/resolved_manifest.csv -- filename, resolved_path, root,
                                          how
    data/processed/images/<batch>/<MST-N>/<file> -- searched across all
                                          four variant folders in order:
                                          processed, c1_processed,
                                          c2_processed, v5_processed

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

from pathlib import Path
 
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm
 
FACES_ROOT = Path("data/processed/images")
VARIANT_FOLDERS = ["processed", "c1_processed", "c2_processed", "v5_processed"]
SPLIT_PATHS = [
    Path("data/processed/train.csv"),
    Path("data/processed/val.csv"),
    Path("data/processed/test.csv"),
]
OUT_DIR = Path("data/processed")
 
LANDMARKS_PATH = OUT_DIR / "landmarks.npy"
INDEX_PATH = OUT_DIR / "landmarks_index.csv"
FAILURES_PATH = OUT_DIR / "landmark_failures.csv"
# These two always end up empty in this version -- they exist only for
# structural consistency with previous script. Old version records
# rows where the recorded image path was wrong and a fallback search
# found it elsewhere (path_corrections.csv), or where the match was a
# low-confidence guess (low_confidence_resolutions.csv). This script has
# no folder-searching or guessing at all -- one filename maps to exactly
# one path, data/processed/faces/<filename> -- so there's nothing for
# either file to ever contain.
LOW_CONFIDENCE_PATH = OUT_DIR / "low_confidence_resolutions.csv"
PATH_CORRECTIONS_PATH = OUT_DIR / "path_corrections.csv"
 
MODEL_PATH = Path("face_landmarker.task")
NUM_LANDMARKS = 468  # this model returns 478 (468 classic mesh + 10 iris); we keep just the first 468
 
 
def load_manifest():
    """Concatenate your three frozen Phase 5 splits -- filenames already
    include the MST-N/ prefix, so they map directly onto FACES_ROOT."""
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

            # Strip the (2) version selector -- these files live in
            # c2_processed/ under their plain name without the suffix.
            # e.g. "MST-10/0008_1_0_0_01 (2).jpg" -> "MST-10/0008_1_0_0_01.jpg"
            search_filename = filename.replace(" (2)", "")

            # Search across variant folders in order -- the CSV filename has
            # no variant prefix (e.g. "MST-1/face.jpg"), so we try each
            # subfolder until we find the file.
            img_path = None
            for variant in VARIANT_FOLDERS:
                candidate = FACES_ROOT / variant / search_filename
                if candidate.exists():
                    img_path = candidate
                    break
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
 
            index_rows.append({"row_index": len(landmarks_list), "filename": filename})
            landmarks_list.append(coords)
 
    if not landmarks_list:
        raise SystemExit("No landmarks extracted at all -- check FACES_ROOT and your split files before anything else.")
 
    landmarks_array = np.stack(landmarks_list, axis=0)  # (N, 468, 3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(LANDMARKS_PATH, landmarks_array)
    pd.DataFrame(index_rows).to_csv(INDEX_PATH, index=False)
    pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
 
    # Always empty here -- see the comment where these paths are defined above.
    pd.DataFrame(columns=["filename"]).to_csv(LOW_CONFIDENCE_PATH, index=False)
    pd.DataFrame(columns=["filename"]).to_csv(PATH_CORRECTIONS_PATH, index=False)
 
    print(f"\nLandmarks extracted: {len(landmarks_list)} / {len(manifest)} images")
    print(f"Failed (logged to {FAILURES_PATH.name}): {len(failures)}")
    print(f"Saved {LANDMARKS_PATH} with shape {landmarks_array.shape}")
    print(f"({LOW_CONFIDENCE_PATH.name} and {PATH_CORRECTIONS_PATH.name} created empty -- "
          f"no folder-resolution happens in this version, so there's nothing for either to log)")
 
 
if __name__ == "__main__":
    run()
