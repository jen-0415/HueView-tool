"""
Phase 7.3 -- Regional Segmentation

Uses Stream A landmarks (7.1) to define 5 region boundaries, applies them
to the SSR-normalized image (7.2) via binary masking.

Includes resolution rules for:
1. Suffix routing e.g. " (2).jpg" -> c2_processed
2. Extension mismatches e.g. manifest .png vs disk .bmp / .jpg
"""

import sys
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_project_root(start: Path, marker: str = "ml_pipeline") -> Path:
    """Walk upward from `start` until a directory containing `marker` is found."""
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
DATA_DIR = PROJECT_ROOT / "ml_pipeline" / "data" / "processed"

LANDMARKS_PATH = DATA_DIR / "landmarks.npy"
INDEX_PATH = DATA_DIR / "landmarks_index.csv"
SSR_ROOT = DATA_DIR / "images_ssr"
OUTPUT_ROOT = DATA_DIR / "regions"
OUTPUT_INDEX_PATH = DATA_DIR / "regions_index.csv"
FAILURES_PATH = DATA_DIR / "region_segmentation_failures.csv"

REGIONS = {
    "forehead": [10, 21, 46, 52, 53, 54, 55, 63, 65, 66, 67, 70, 103, 105, 107, 109,
                 162, 251, 276, 282, 283, 284, 285, 293, 295, 296, 297, 300, 332, 334, 336, 338],
    "left_cheek": [7, 48, 64, 97, 98, 115, 133, 137, 138, 144, 145, 153, 154, 155,
                   163, 177, 215, 220, 227],
    "right_cheek": [249, 264, 278, 294, 326, 327, 344, 362, 366, 373, 374, 380, 381,
                    382, 390, 401, 435, 440, 447],
    "nose_bridge": [1, 2, 4, 5, 6, 19, 45, 94, 168, 195, 197, 275],
    "jawline": [136, 148, 149, 150, 152, 172, 176, 365, 377, 378, 379, 397, 400],
}

_clashes = {
    f"{a} + {b}": sorted(set(REGIONS[a]) & set(REGIONS[b]))
    for a, b in combinations(REGIONS, 2)
    if set(REGIONS[a]) & set(REGIONS[b])
}
if _clashes:
    raise ValueError(f"REGIONS index groups overlap: {_clashes}.")


def build_region_mask(landmarks_px, region_indices, shape):
    pts = landmarks_px[region_indices, :2].astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 1)
    return mask


def get_filename_variants(filename):
    """Generates path variations considering clean names and extension mismatches (.png -> .bmp/.jpg)."""
    clean_filename = re.sub(r"\s*\(\d+\)(?=\.[^.]+$)", "", filename)
    
    base_names = [filename, clean_filename]
    variants = []

    for name in base_names:
        p = Path(name)
        variants.append(name)
        # Alternate extensions if original is .png or .jpg
        for ext in [".bmp", ".jpg", ".jpeg", ".png"]:
            if p.suffix.lower() != ext:
                variants.append(str(p.with_suffix(ext)))
                
    return list(dict.fromkeys(variants))  # Preserve order, remove duplicates


def locate_ssr_image(filename):
    """Locates the SSR image across cohort directories handling suffixes and extension swaps."""
    variants = get_filename_variants(filename)

    # 1. Check direct path with variations
    for v in variants:
        direct_path = SSR_ROOT / v
        if direct_path.exists():
            return direct_path

    # Determine cohort priority order
    suffix_match = re.search(r"\s*\((\d+)\)(?=\.[^.]+$)", filename)
    if suffix_match:
        suffix_num = suffix_match.group(1)
        priority_cohorts = [f"c{suffix_num}_processed", "v5_processed", "c1_processed", "processed"]
    else:
        priority_cohorts = ["processed", "v5_processed", "c1_processed", "c2_processed"]

    # 2. Check priority cohorts
    for cohort in priority_cohorts:
        for v in variants:
            candidate = SSR_ROOT / cohort / v
            if candidate.exists():
                return candidate

            # Handle duplicated MST folder prefix inside cohort directory
            v_path = Path(v)
            candidate_sub = SSR_ROOT / cohort / v_path
            if candidate_sub.exists():
                return candidate_sub

    # 3. Fallback check across all subfolders in SSR_ROOT
    for cohort_dir in SSR_ROOT.iterdir():
        if cohort_dir.is_dir():
            for v in variants:
                cand = cohort_dir / v
                if cand.exists():
                    return cand

                if v.startswith(f"{cohort_dir.name}/"):
                    stripped = v.replace(f"{cohort_dir.name}/", "", 1)
                    if (cohort_dir / stripped).exists():
                        return cohort_dir / stripped

    return None


def process_one(args):
    filename, row_index, landmarks_px = args

    ssr_path = locate_ssr_image(filename)
    if ssr_path is None:
        return (filename, None, "ssr_image_missing_or_unreadable")

    ssr_image = cv2.imread(str(ssr_path))
    if ssr_image is None:
        return (filename, None, "ssr_image_missing_or_unreadable")

    basename = Path(filename).stem
    mst_folder = Path(filename).parent
    out_dir = OUTPUT_ROOT / mst_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for region_name, region_indices in REGIONS.items():
        mask = build_region_mask(landmarks_px, region_indices, ssr_image.shape)
        masked_image = ssr_image.copy()
        masked_image[mask == 0] = 0

        out_path = out_dir / f"{basename}_{region_name}.jpg"
        cv2.imwrite(str(out_path), masked_image)
        outputs.append((filename, region_name, str(out_path)))

    return (filename, outputs, None)


def run():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reading landmarks from: {LANDMARKS_PATH}")

    landmarks_array = np.load(LANDMARKS_PATH)  # (N, 468, 3)
    index_df = pd.read_csv(INDEX_PATH)
    print(f"Loaded {len(index_df)} landmark rows, landmarks array shape {landmarks_array.shape}")

    jobs = [
        (row["filename"], row["row_index"], landmarks_array[row["row_index"]])
        for _, row in index_df.iterrows()
    ]

    output_rows = []
    failures = []

    with Pool(processes=cpu_count()) as pool:
        for filename, outputs, error in tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs), desc="Phase 7.3"):
            if error is not None:
                failures.append({"filename": filename, "reason": error})
            else:
                for fname, region, path in outputs:
                    output_rows.append({"filename": fname, "region": region, "output_path": path})

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(OUTPUT_INDEX_PATH, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)

    n_success = len(index_df) - len(failures)
    print(f"\nProcessed {n_success} / {len(index_df)} images successfully")
    print(f"Failed: {len(failures)} (logged to {FAILURES_PATH.name})" if failures else "Failed: 0")
    print(f"Generated {len(output_rows)} region-masked images ({n_success} images x 5 regions)")
    print(f"Saved index to {OUTPUT_INDEX_PATH}")


if __name__ == "__main__":
    run()