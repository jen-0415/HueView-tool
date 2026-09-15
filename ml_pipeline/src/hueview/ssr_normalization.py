"""
SSR (Single-Scale Retinex) Illumination Normalization
Phase 7, Stream B (parallel to 7.1) -- HueView

R(x,y) = log(I(x,y)) - log(F(x,y) * I(x,y)), F = Gaussian(sigma=80)
Applied independently per channel, rescaled to 0-255. The SSR formula
itself is UNCHANGED from the original.

Job list comes from landmarks_index.csv (Phase 7.1's output) rather than
scanning a folder -- that file already contains exactly the images that
successfully got landmarks, so there's no point re-deriving "which images
have landmarks" here, and no point SSR-normalizing an image that has none
to build regions from in 7.3.

Each filename is resolved to its real file across the four batch folders
(processed/, c1_processed/, c2_processed/, v5_processed/) via
path_resolver.resolve_image_path -- the same resolver every other script
uses -- instead of assuming a single flat data/processed/faces/ folder,
which doesn't exist in this dataset layout.

Output mirrors your MST-N/ structure under data/processed/faces_ssr/.

Paths are anchored to <project_root> (the folder containing "data/"), found
by walking up from this file's own location -- works the same whether you
run it from the project root, from src/hueview, or via an IDE "run" button.
"""

import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------ project root --


def find_project_root(start: Path, marker: str = "data") -> Path:
    """Walk upward from `start` until a directory containing `marker` is found."""
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start  # fallback: marker not found anywhere above `start`


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)

# path_resolver.py lives under src/baseline/ -- add src/ to the path so this
# script can import it regardless of where it's run from.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ml_pipeline.src.baseline.path_resolver import resolve_image_path  # noqa: E402

# ---------------------------------------------------------------- config --

LANDMARKS_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "faces_ssr"

SIGMA = 80.0
EPSILON = 1.0                      # avoids log(0); standard in Retinex implementations

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


# ------------------------------------------------------------- job list --


def collect_jobs():
    """Job list = every image landmark_extraction.py successfully processed
    (landmarks_index.csv already excludes failures -- no separate exclusion
    list needed here). Each filename is resolved to its real file across
    the 4 batch folders via path_resolver.resolve_image_path, so this reads
    identical bytes to every other script instead of assuming a flat
    faces/ folder.

    Returns (jobs, unresolved) where jobs is a list of (src_path, dst_path)
    tuples and unresolved is a list of filenames path_resolver couldn't
    find anywhere.
    """
    if not LANDMARKS_INDEX_PATH.is_file():
        raise SystemExit(
            f"ERROR: {LANDMARKS_INDEX_PATH} not found -- run landmark_extraction.py (7.1) first."
        )

    index_df = pd.read_csv(LANDMARKS_INDEX_PATH)
    jobs = []
    unresolved = []

    for filename in index_df["filename"]:
        src_path, rule, root = resolve_image_path(filename)
        if src_path is None:
            unresolved.append(filename)
            continue
        jobs.append((src_path, OUTPUT_ROOT / filename))

    return jobs, unresolved


def main():
    print(f"Project root: {PROJECT_ROOT}")

    all_jobs, unresolved = collect_jobs()
    print(f"{LANDMARKS_INDEX_PATH.name}: {len(all_jobs)} resolved, {len(unresolved)} unresolved")

    if unresolved:
        log_path = PROJECT_ROOT / "ssr_unresolved.log"
        log_path.write_text("\n".join(unresolved))
        print(f"Unresolved filenames logged to {log_path}")

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
