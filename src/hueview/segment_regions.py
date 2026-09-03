"""
Phase 7.3 -- Regional Segmentation

Uses Stream A landmarks (7.1) to define 5 region boundaries, applies them
to the SSR-normalized image (7.2) via binary masking. Landmarks come from
the raw image; masking happens on the SSR-normalized image, per spec.

Region index groups are the ones confirmed from derive_regions.py's v3 run
(forehead/nose_bridge/jawline solidly anchored on official MediaPipe
groups; left_cheek/right_cheek approximated and visually verified against
a real photo).

Input:
    data/processed/landmarks.npy         -- (N, 468, 3) pixel coords, from 7.1
    data/processed/landmarks_index.csv   -- row_index, filename
    data/processed/faces_ssr/<filename>  -- SSR-normalized images, from 7.2

Output:
    data/processed/regions/<MST-N>/<basename>_<region>.jpg -- one masked
        image per region per face (5 per successful image), zeros outside
        the region, per spec
    data/processed/regions_index.csv     -- filename, region, output_path
"""

from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

LANDMARKS_PATH = Path("data/processed/landmarks.npy")
INDEX_PATH = Path("data/processed/landmarks_index.csv")
SSR_ROOT = Path("data/processed/faces_ssr")
OUTPUT_ROOT = Path("data/processed/regions")
OUTPUT_INDEX_PATH = Path("data/processed/regions_index.csv")

# Confirmed region index groups -- see derive_regions.py / region_reference.png
REGIONS = {
    "forehead": [10, 21, 46, 52, 53, 54, 55, 63, 65, 66, 67, 70, 103, 105, 107, 109,
                 251, 276, 282, 283, 284, 285, 293, 295, 296, 297, 300, 332, 334, 336, 338],
    "left_cheek": [2, 5, 6, 19, 45, 48, 64, 93, 94, 97, 98, 115, 127, 132, 195, 197,
                   220, 234, 249, 362, 373, 374, 380, 381, 382, 390],
    "right_cheek": [4, 7, 133, 144, 145, 153, 154, 155, 163, 275, 278, 294, 323, 326,
                     327, 344, 356, 361, 440, 454],
    "nose_bridge": [1, 2, 4, 5, 6, 19, 45, 94, 168, 195, 197, 275],
    "jawline": [58, 136, 148, 149, 150, 152, 172, 176, 288, 365, 377, 378, 379, 397, 400],
}


def build_region_mask(landmarks_px, region_indices, shape):
    """landmarks_px: (468, 3) array [x_px, y_px, z_rel]. Returns a binary
    mask (h, w), 1 inside the region's convex hull, 0 outside."""
    pts = landmarks_px[region_indices, :2].astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 1)
    return mask


def process_one(args):
    filename, row_index, landmarks_px = args
    ssr_path = SSR_ROOT / filename

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
        for filename, outputs, error in tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs)):
            if error is not None:
                failures.append({"filename": filename, "reason": error})
            else:
                for fname, region, path in outputs:
                    output_rows.append({"filename": fname, "region": region, "output_path": path})

    pd.DataFrame(output_rows).to_csv(OUTPUT_INDEX_PATH, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(Path("data/processed/region_segmentation_failures.csv"), index=False)

    n_success = len(index_df) - len(failures)
    print(f"\nProcessed {n_success} / {len(index_df)} images successfully")
    print(f"Failed: {len(failures)} (logged to region_segmentation_failures.csv)" if failures else "Failed: 0")
    print(f"Generated {len(output_rows)} region-masked images ({n_success} images x 5 regions)")
    print(f"Saved index to {OUTPUT_INDEX_PATH}")


if __name__ == "__main__":
    run()