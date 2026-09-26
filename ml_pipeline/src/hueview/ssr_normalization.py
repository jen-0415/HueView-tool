"""
SSR (Single-Scale Retinex) Illumination Normalization
Phase 7, Stream B (parallel to 7.1) -- HueView
Manuscript: Chapter 3, Stage 2 (SSR Illumination Normalization); Data
Generation (Feature Selection).

R(x,y) = log(I(x,y)) - log(F(x,y) * I(x,y)), F = Gaussian(sigma=30)
Applied independently per channel ("SSR is applied independently to the Red,
Green, and Blue channels", Data Generation), rescaled to 0-255. Runs on the
full, unmasked 224x224 face, before any regional masking.

SIGMA = 30 (was 80). sigma = 80 flattened the skin colour channels (a*/b*
dropped from ~10-15 to ~2-5). The manuscript still says 80 in five places
(Definition of Terms x2, Stage 2, Data Generation, Appendix 1) -- update them.

Job list comes from landmarks_index.csv (Phase 7.1's output) rather than
scanning a folder -- that file already contains exactly the images that
successfully got landmarks, so there's no point SSR-normalizing an image that
has none to build regions from in 7.3.

Input: data/processed/images/<filename> -- the MST-N folder rebuilt from
resolved_manifest.csv (rebuild_images_from_manifest.py), where every manifest
filename is the exact image the manifest means. Read by exact path, so the
result no longer depends on which folder the script is run from.

Output mirrors the MST-N/ structure under data/processed/images_ssr/ -- the
folder Phases 7.3 (segment_regions.py), 7.4 (run_phase7_4_batch.py),
7.5 (run_phase7_5_batch.py) and 9 (run_phase9_batch.py) read.

Paths are anchored to <project_root> (the folder containing "data/", i.e.
ml_pipeline/), found by walking up from this file's own location -- works the
same whether you run it from the repo root, from src/hueview, or via an IDE.
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

# ---------------------------------------------------------------- config --

LANDMARKS_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
IMAGES_ROOT = PROJECT_ROOT / "data" / "processed" / "images"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "images_ssr"

SIGMA = 30.0
EPSILON = 1.0                      # avoids log(0); standard in Retinex implementations

SAMPLE_SIZE = None                 # e.g. 20 to sanity-check output before the full run

# ------------------------------------------------------------------- SSR --


def apply_ssr(image: np.ndarray, sigma: float = SIGMA, epsilon: float = EPSILON) -> np.ndarray:
    """Single-Scale Retinex, applied independently to each channel."""
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
    (landmarks_index.csv already excludes failures). Each filename is read
    from IMAGES_ROOT/<filename> by exact path.

    Returns (jobs, unresolved) where jobs is a list of (src_path, dst_path)
    tuples and unresolved is a list of filenames with no file at that path.
    """
    if not LANDMARKS_INDEX_PATH.is_file():
        raise SystemExit(
            f"ERROR: {LANDMARKS_INDEX_PATH} not found -- run landmark_extraction.py (7.1) first."
        )
    if not IMAGES_ROOT.is_dir():
        raise SystemExit(f"ERROR: {IMAGES_ROOT} not found.")

    index_df = pd.read_csv(LANDMARKS_INDEX_PATH)
    jobs = []
    unresolved = []

    for filename in index_df["filename"]:
        rel = str(filename).replace("\\", "/")
        src_path = IMAGES_ROOT / rel
        if not src_path.is_file():
            unresolved.append(rel)
            continue
        jobs.append((src_path, OUTPUT_ROOT / rel))

    return jobs, unresolved


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"SSR sigma = {SIGMA}  (per channel)")
    print(f"Input : {IMAGES_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")

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
