"""
SSR (Single-Scale Retinex) Illumination Normalization
Phase 7, Stream B (parallel to 7.1) -- HueView

R(x,y) = log(I(x,y)) - log(F(x,y) * I(x,y)), F = Gaussian(sigma=80)
Applied independently per channel, rescaled to 0-255. The SSR formula
itself is UNCHANGED from the original -- only the file-finding logic
around it was adapted, same as landmark_extraction.py.
 
Images listed in landmark_failures.csv are skipped -- no point SSR-
normalizing an image that has no landmarks to build regions from in 7.3.
 
Output mirrors your MST-N/ structure under data/processed/faces_ssr/.

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

FACES_PATH = PROJECT_ROOT / "data" / "processed" / "faces"
FAILURES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "landmark_failures.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "faces_ssr"
 
SIGMA = 80.0
EPSILON = 1.0                      # avoids log(0); standard in Retinex implementations
VALID_EXTS = {".jpg", ".jpeg", ".png"}  # no .bmp -- those were already converted back in Phase 3
 
SAMPLE_SIZE = None                 # e.g. 20 to sanity-check output before the full run

# ------------------------------------------------------------------- SSR --


def apply_ssr(image: np.ndarray, sigma: float = SIGMA, epsilon: float = EPSILON) -> np.ndarray:
    """Single-Scale Retinex, applied independently to each channel. Unchanged
    from the original -- this formula was never the part that was broken."""
    img = image.astype(np.float64) + epsilon
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
    """Read landmark_failures.csv (filename, reason columns -- your own
    script's format) and return the set of relative filenames to skip,
    e.g. 'MST-3/img.jpg'."""
    excluded = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = (row.get("filename") or "").strip()
            if value:
                excluded.add(value)
    return excluded


# ------------------------------------------------------------- job list --


def collect_jobs(excluded: set) -> tuple:
    """Find every image under data/processed/faces/, skipping filenames
    listed in landmark_failures.csv. Returns (jobs, n_skipped)."""
    all_images = [p for p in FACES_PATH.rglob("*") if p.suffix.lower() in VALID_EXTS]
    jobs = []
    n_skipped = 0
    for p in all_images:
        relative = str(p.relative_to(FACES_PATH)).replace("\\", "/")  # e.g. "MST-3/img.jpg"
        if relative in excluded:
            n_skipped += 1
            continue
        jobs.append((p, OUTPUT_ROOT / p.relative_to(FACES_PATH)))
    return jobs, n_skipped
 
 
def main():
    print(f"Project root: {PROJECT_ROOT}")
 
    if not FACES_PATH.is_dir():
        raise SystemExit(f"ERROR: {FACES_PATH} not found -- check your folder structure before running.")
 
    excluded = load_excluded_filenames(FAILURES_CSV_PATH) if FAILURES_CSV_PATH.is_file() else set()
    if excluded:
        print(f"{FAILURES_CSV_PATH.name}: {len(excluded)} filenames to exclude "
              f"(no landmarks -> can't build regions for these anyway)")
 
    all_jobs, n_skipped = collect_jobs(excluded)
    print(f"{FACES_PATH.name}: {len(all_jobs)} images to process ({n_skipped} skipped via landmark_failures.csv)")
 
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