"""
Manual Phase 7.4 QA -- HSV skin filtering validated against REAL Phase
7.1-7.3 output.

run_region_qa.py's fallback-detection looks for src/hueview/ssr.py and
src/hueview/segmentation.py, which don't exist under those names, and even
if they did, its call site doesn't pass landmarks through to
segment_regions() the way its own docstring says it should. So as it
stands, it validates Table-3 rectangle approximations, not your actual
v5-corrected landmark polygons.

This script bypasses that entirely: it recomputes the real geometric masks
directly from landmarks.npy using segment_regions.py's own REGIONS dict and
build_region_mask() -- the exact same masks 7.3 wrote to disk -- loads the
real faces_ssr/ output, converts BGR->RGB (hsv_skin_filter.py requires RGB;
your saved faces_ssr/ files are BGR), and runs everything through the real
hsv_skin_filter.filter_all_regions(). The retention numbers and contact
sheets this produces are the genuine Phase 7.4 checkpoint.

Usage:
    python qa_hsv_filter.py --n 12
    python qa_hsv_filter.py --n 12 --hsv-source original
    python qa_hsv_filter.py --filenames "MST-1/104_Brazilian Faces104-01_face_1.jpg"
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ------------------------------------------------------------ project root --


def find_project_root(start: Path, marker: str = "data") -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_pipeline.src.baseline.path_resolver import resolve_image_path  # noqa: E402
from ml_pipeline.src.hueview.segment_regions import REGIONS, build_region_mask  # noqa: E402
from ml_pipeline.src.hueview.hsv_skin_filter import FilterConfig, filter_all_regions, summarize  # noqa: E402
from ml_pipeline.src.hueview.regions import REGION_ORDER  # noqa: E402

# ---------------------------------------------------------------- config --

LANDMARKS_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
SSR_ROOT = PROJECT_ROOT / "data" / "processed" / "faces_ssr"
OUT_DIR = PROJECT_ROOT / "results" / "phase7_4_qa"

REGION_COLORS_BGR = {
    "forehead":    (71, 99, 255),
    "left_cheek":  (113, 179, 60),
    "right_cheek": (225, 105, 65),
    "nose_bridge": (0, 215, 255),
    "jawline":     (211, 85, 186),
}

# --------------------------------------------------------------- helpers --


def label(img: np.ndarray, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    strip = np.full((22, w, 3), 245, dtype=np.uint8)
    cv2.putText(strip, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([img, strip])


def overlay_masks(base_bgr: np.ndarray, masks_dict) -> np.ndarray:
    out = base_bgr.astype(np.float32).copy()
    for name, mask in masks_dict.items():
        if mask is None or not mask.any():
            continue
        color = np.array(REGION_COLORS_BGR.get(name, (255, 255, 255)), dtype=np.float32)
        out[mask] = 0.55 * out[mask] + 0.45 * color
    return np.clip(out, 0, 255).astype(np.uint8)


def build_geometric_masks(landmarks_px: np.ndarray, shape) -> dict:
    """The exact same masks segment_regions.py (7.3) computes and writes to
    disk -- reused here directly rather than re-derived from saved crops."""
    return {name: build_region_mask(landmarks_px, idx, shape).astype(bool) for name, idx in REGIONS.items()}


def process_one(filename, row_index, landmarks_array, config, write_sheet):
    ssr_path = SSR_ROOT / filename
    ssr_bgr = cv2.imread(str(ssr_path))
    if ssr_bgr is None:
        print(f"[skip] SSR image missing: {ssr_path}")
        return None

    # hsv_skin_filter.py requires RGB -- faces_ssr/ files are BGR on disk.
    ssr_rgb = cv2.cvtColor(ssr_bgr, cv2.COLOR_BGR2RGB)

    landmarks_px = landmarks_array[row_index]
    geometric_masks = build_geometric_masks(landmarks_px, ssr_rgb.shape)

    reference_rgb = None
    orig_bgr = None
    orig_path, _, _ = resolve_image_path(filename)
    if orig_path:
        orig_bgr = cv2.imread(str(orig_path))
    if orig_bgr is not None:
        reference_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    if config.hsv_source == "original" and reference_rgb is None:
        print(f"[skip] original crop not found for {filename}, needed for hsv_source='original'")
        return None

    patches = filter_all_regions(
        ssr_image=ssr_rgb,
        geometric_masks=geometric_masks,
        config=config,
        reference_image=reference_rgb,
    )
    stats = summarize(patches)

    if write_sheet:
        if orig_bgr is None:
            orig_bgr = np.zeros_like(ssr_bgr)
        geom_overlay = overlay_masks(ssr_bgr, geometric_masks)
        skin_masks = {n: p.skin_mask for n, p in patches.items()}
        skin_overlay = overlay_masks(ssr_bgr, skin_masks)

        avg_retention = float(np.mean([s["retention"] for s in stats.values()]))
        panels = [
            label(orig_bgr, "original"),
            label(ssr_bgr, "SSR"),
            label(geom_overlay, "7.3 polygons"),
            label(skin_overlay, f"7.4 skin (avg ret={avg_retention:.0%})"),
        ]
        sheet = np.hstack(panels)
        safe_name = Path(filename).stem.replace("/", "_") + "_hsv_qa.png"
        cv2.imwrite(str(OUT_DIR / safe_name), sheet)

    return [{"filename": filename, "region": region, **s} for region, s in stats.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="random sample size")
    ap.add_argument("--filenames", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hsv-source", choices=["ssr", "original"], default="ssr")
    ap.add_argument("--no-sheets", action="store_true")
    args = ap.parse_args()

    if not LANDMARKS_PATH.is_file() or not INDEX_PATH.is_file():
        raise SystemExit("ERROR: landmarks.npy / landmarks_index.csv not found -- run landmark_extraction.py (7.1) first.")

    landmarks_array = np.load(LANDMARKS_PATH)
    index_df = pd.read_csv(INDEX_PATH)
    filename_to_row = dict(zip(index_df["filename"], index_df["row_index"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.filenames:
        chosen = args.filenames
    else:
        random.seed(args.seed)
        chosen = random.sample(list(filename_to_row.keys()), min(args.n, len(filename_to_row)))

    config = FilterConfig(hsv_source=args.hsv_source)
    print(f"hsv_source = {args.hsv_source}")
    print(f"Sampling {len(chosen)} image(s)\n")

    all_rows = []
    for filename in chosen:
        row_index = filename_to_row.get(filename)
        if row_index is None:
            print(f"[skip] {filename} not found in landmarks_index.csv")
            continue
        rows = process_one(filename, row_index, landmarks_array, config, write_sheet=not args.no_sheets)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("[error] nothing processed")
        return

    stats_df = pd.DataFrame(all_rows)
    stats_csv = OUT_DIR / "phase7_4_coverage_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    print("RETENTION BY REGION (fraction of polygon surviving HSV filtering)")
    print("-" * 60)
    pivot = stats_df.pivot_table(index="region", values="retention", aggfunc="mean").reindex(REGION_ORDER)
    print(pivot.round(3).to_string())

    print("\nFILTER TIER USAGE (status counts per region)")
    print("-" * 60)
    tier = stats_df.groupby(["region", "status"]).size().unstack(fill_value=0)
    print(tier.reindex(REGION_ORDER).to_string())

    print(f"\nWrote {stats_csv}")
    if not args.no_sheets:
        print(f"Wrote contact sheets to {OUT_DIR}/")

    print("\nWHAT TO LOOK FOR")
    print("  - retention consistently below ~0.4 on a region across most images")
    print("    means the thresholds are miscalibrated for your data, not that")
    print("    those particular faces are unusual.")
    print("  - lots of 'relaxed' or 'geometry_fallback' status is the same signal --")
    print("    try --hsv-source original and compare retention side by side.")
    print("  - in the sheets: panel 4 should have eyebrows/lips/hair gone, with")
    print("    real cheek/forehead skin still intact.")


if __name__ == "__main__":
    main()
