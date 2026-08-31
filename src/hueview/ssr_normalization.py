"""
SSR (Single-Scale Retinex) Illumination Normalization
Phase 7, Stream B (parallel to 7.1) -- HueView

R(x,y) = log(I(x,y)) - log(F(x,y) * I(x,y)), F = Gaussian(sigma=80)
Applied independently per channel, rescaled to 0-255, saved for the
downstream regional segmentation step.

Expected layout:
    <project_root>/data/processed/images/
        processed/MST-1/...MST-10/*.jpg
        c1_processed/MST-1/...MST-10/*.jpg
        c2_processed/MST-1/...MST-10/*.jpg
        v5_processed/MST-1/...MST-10/*.jpg
    <project_root>/data/processed/landmark_failures.csv   (optional -- listed images are skipped)

Output mirrors the same cohort/MST-N structure under data/processed/images_ssr/.

Paths are anchored to <project_root> (the folder containing "data/"), found
by walking up from this file's own location -- works the same whether you
run it from the project root, from src/hueview, or via an IDE "run" button.
"""

import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------ project root --


def find_project_root(start: Path, marker: str = "data") -> Path:
    """Walk upward from `start` until a directory containing `marker` is found."""
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start  # fallback: marker not found anywhere above `start`


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)

# ---------------------------------------------------------------- config --

IMAGES_PATH = PROJECT_ROOT / "data" / "processed" / "images"
FAILURES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "landmark_failures.csv"
OUTPUT_ROOT = IMAGES_PATH.parent / "images_ssr"  # data/processed/images_ssr

DATASET_PATH = IMAGES_PATH / "processed"
C1_PATH = IMAGES_PATH / "c1_processed"
C2_PATH = IMAGES_PATH / "c2_processed"
V5_PATH = IMAGES_PATH / "v5_processed"
INPUT_ROOTS = [DATASET_PATH, C1_PATH, C2_PATH, V5_PATH]

SIGMA = 80.0
EPSILON = 1.0                     # avoids log(0); standard in Retinex implementations
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

SAMPLE_SIZE = None                 # e.g. 20 to sanity-check output before the full run

FILENAME_COLUMN_CANDIDATES = {"filename", "file", "file_name", "image", "image_name", "img", "path", "image_path"}

# ------------------------------------------------------------------- SSR --


def apply_ssr(image: np.ndarray, sigma: float = SIGMA, epsilon: float = EPSILON) -> np.ndarray:
    """Single-Scale Retinex, applied independently to each channel."""
    img = image.astype(np.float64) + epsilon  # avoid log(0)
    out = np.empty_like(img)

    for c in range(img.shape[2]):
        channel = img[:, :, c]
        illumination = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
        retinex = np.log(channel) - np.log(illumination)

        r_min, r_max = retinex.min(), retinex.max()
        span = r_max - r_min
        out[:, :, c] = 0.0 if span < 1e-6 else (retinex - r_min) / span * 255.0

    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- worker --


def process_one(paths):
    src_path, dst_path = paths
    try:
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            return (str(src_path), "unreadable")
        result = apply_ssr(img)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(dst_path), result)
        if not ok:
            return (str(src_path), "write failed")
        return None
    except Exception as e:
        return (str(src_path), repr(e))


# --------------------------------------------------------- failures csv --


def load_excluded_filenames(csv_path: Path) -> set:
    """Read landmark_failures.csv and return the set of filenames (basenames) to skip."""
    excluded = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        matches = [c for c in fieldnames if c.strip().lower().replace(" ", "_") in FILENAME_COLUMN_CANDIDATES]
        col = matches[0] if matches else (fieldnames[0] if fieldnames else None)
        if col is None:
            return excluded
        print(f"  using column '{col}' in {csv_path.name}")
        for row in reader:
            value = (row.get(col) or "").strip()
            if value:
                excluded.add(Path(value).name)
    return excluded


# ------------------------------------------------------------- job list --


def collect_jobs(cohort_root: Path, excluded: set) -> tuple:
    """Find image jobs under one cohort folder (e.g. .../images/c1_processed).
    Returns (jobs, n_skipped_via_csv)."""
    all_images = [p for p in cohort_root.rglob("*") if p.suffix.lower() in VALID_EXTS]
    jobs = [
        (p, OUTPUT_ROOT / p.relative_to(IMAGES_PATH))
        for p in all_images
        if p.name not in excluded
    ]
    n_skipped = len(all_images) - len(jobs)
    return jobs, n_skipped


def main():
    print(f"Project root: {PROJECT_ROOT}")

    excluded = load_excluded_filenames(FAILURES_CSV_PATH) if FAILURES_CSV_PATH.is_file() else set()
    if excluded:
        print(f"{FAILURES_CSV_PATH}: {len(excluded)} filenames to exclude")

    all_jobs = []
    for root in INPUT_ROOTS:
        if not root.is_dir():
            print(f"WARNING: {root} not found -- skipping")
            continue
        jobs, n_skipped = collect_jobs(root, excluded)
        print(f"{root.name}: {len(jobs)} images to process ({n_skipped} skipped via landmark_failures.csv)")
        all_jobs.extend(jobs)

    if SAMPLE_SIZE:
        all_jobs = all_jobs[:SAMPLE_SIZE]
        print(f"SAMPLE_SIZE set -- only processing {len(all_jobs)} images")

    print(f"Total: {len(all_jobs)} images, {cpu_count()} workers")

    failures = []
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_one, all_jobs), total=len(all_jobs)):
            if result is not None:
                failures.append(result)

    print(f"Finished. {len(failures)} failed out of {len(all_jobs)}.")
    if failures:
        log_path = PROJECT_ROOT / "ssr_failures.log"
        with open(log_path, "w") as f:
            for path, err in failures:
                f.write(f"{path}\t{err}\n")
        print(f"Failures logged to {log_path}")


if __name__ == "__main__":
    main()