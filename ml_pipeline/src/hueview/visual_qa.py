"""
Phase 7 Visual Inspection QA Script

Overlay 5 regional geometric masks and HSV skin masks onto a random sample
of images to visually verify anatomical placement and HSV filtering accuracy.

Panel 2 = geometric region polygons (7.3); panel 3 = only the skin pixels kept
by the HSV filter (7.4). Label maps are read by exact manifest filename.

Outputs:
    results/qa_visualizations/sample_<filename>.jpg
"""

import os
import sys
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def find_project_root(start: Path, marker: str = "ml_pipeline") -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.src.hueview.regions import decode_label_map, REGION_ORDER

# ---------------------------------------------------------------- Config --

DATA_PROCESSED = PROJECT_ROOT / "ml_pipeline" / "data" / "processed"
INDEX_PATH = DATA_PROCESSED / "landmarks_index.csv"
LABEL_MAPS_ROOT = DATA_PROCESSED / "label_maps"
SSR_ROOT = DATA_PROCESSED / "images_ssr"
OUTPUT_QA_DIR = PROJECT_ROOT / "results" / "qa_visualizations"

NUM_SAMPLES = 20  # Number of random test images to visually inspect

# Color palette (BGR) for the 5 regions
REGION_COLORS = {
    "forehead": (0, 0, 255),       # Red
    "left_cheek": (0, 255, 0),     # Green
    "right_cheek": (255, 0, 0),    # Blue
    "nose_bridge": (0, 255, 255),  # Yellow
    "jawline": (255, 0, 255)       # Magenta
}


def resolve_ssr_path(filename: str) -> Path | None:
    str_file = str(filename).replace("\\", "/")
    if "images/" in str_file:
        str_file = str_file.split("images/")[-1]

    cand = SSR_ROOT / str_file
    if cand.is_file():
        return cand

    for cohort in ["processed", "v5_processed", "c1_processed", "c2_processed"]:
        cand_c = SSR_ROOT / cohort / str_file
        if cand_c.is_file():
            return cand_c

    bare_name = Path(str_file).name.lower()
    matched = list(SSR_ROOT.rglob(bare_name))
    return matched[0] if matched and matched[0].is_file() else None


def generate_qa_overlay(filename: str):
    str_file = str(filename).replace("\\", "/")
    if "images/" in str_file:
        str_file = str_file.split("images/")[-1]

    label_path = LABEL_MAPS_ROOT / Path(str_file).with_suffix(".png")
    ssr_path = resolve_ssr_path(filename)

    if not label_path.is_file() or ssr_path is None:
        return False

    label_map = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    ssr_bgr = cv2.imread(str(ssr_path))

    if label_map is None or ssr_bgr is None:
        return False

    # Panel 1: Original SSR Image
    panel1_orig = ssr_bgr.copy()

    # Panel 2: Color-Coded Mask Overlays
    panel2_overlay = ssr_bgr.copy()
    color_mask = np.zeros_like(ssr_bgr, dtype=np.uint8)

    # Decode label map patches
    ssr_rgb = cv2.cvtColor(ssr_bgr, cv2.COLOR_BGR2RGB)
    patches = decode_label_map(label_map, ssr_rgb)

    # Panel 3: HSV Filtered Skin Pixels Isolations
    panel3_filtered = np.zeros_like(ssr_bgr)

    for region in REGION_ORDER:
        if region in patches and patches[region] is not None:
            patch = patches[region]
            
            if patch.image is not None:
                # Panel 2: the region's geometric polygon (Phase 7.3).
                geom = patch.geometric_mask
                # Panel 3: only the pixels that SURVIVED the HSV filter
                # (Phase 7.4). patch.image holds the whole polygon, so the
                # skin mask must be used here to show what HSV removed.
                skin = patch.skin_mask if patch.skin_mask is not None else geom

                if geom.any():
                    color = REGION_COLORS.get(region, (255, 255, 255))
                    color_mask[geom] = color
                if skin.any():
                    panel3_filtered[skin] = ssr_bgr[skin]

    # Blend panel 2 (40% color mask overlay)
    cv2.addWeighted(color_mask, 0.4, panel2_overlay, 0.6, 0, panel2_overlay)

    # Stitch panels horizontally: [Original SSR | Color Overlay | HSV Filtered Skin]
    h, w = ssr_bgr.shape[:2]
    combined_grid = np.hstack([panel1_orig, panel2_overlay, panel3_filtered])

    # Add text labels on top
    cv2.putText(combined_grid, "1. SSR Image", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined_grid, "2. Mask Overlays", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined_grid, "3. HSV Filtered Skin", (2 * w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save output QA image
    OUTPUT_QA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_QA_DIR / f"qa_{Path(str_file).stem}.jpg"
    cv2.imwrite(str(out_file), combined_grid)
    return True


def run():
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(f"Missing {INDEX_PATH}")

    df = pd.read_csv(INDEX_PATH)
    sample_filenames = df["filename"].sample(n=min(NUM_SAMPLES, len(df)), random_state=42).tolist()

    print(f"Generating visual QA check for {len(sample_filenames)} sample images...")
    successful = 0

    for filename in sample_filenames:
        if generate_qa_overlay(filename):
            successful += 1

    print(f"\nSaved {successful} visual QA images to: {OUTPUT_QA_DIR}")


if __name__ == "__main__":
    run()