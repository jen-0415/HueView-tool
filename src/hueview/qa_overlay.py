"""
qa_overlay.py -- Manual visual QA for real Phase 7.3 region masks.

Builds a 3-panel contact sheet per sampled image: original crop | SSR
(faces_ssr/) | all 5 real region masks overlaid in color on the SSR image.

Unlike run_region_qa.py, this reads DIRECTLY from your actual generated
output -- regions_index.csv, faces_ssr/<filename>, and
regions/<MST-N>/<basename>_<region>.jpg -- with no fallback logic and no
dependency on src/hueview/ssr.py or src/hueview/segmentation.py (neither
of which exists under those names). What you see here is exactly what
Phase 7.2/7.3 actually produced, nothing reconstructed or approximated.

Region masks are recovered from the saved region crop images themselves
(a pixel counts as "in the region" if it's non-black) -- this is a fair
approximation for eyeballing since Phase 7.3 zeros everything outside
each region's convex hull, and true black pixels inside a real skin
region are rare after SSR's brightening/flattening effect.

Usage:
    python qa_overlay.py --n 10
    python qa_overlay.py --filenames "MST-1/104_Brazilian Faces104-01_face_1.jpg"
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
from baseline.path_resolver import resolve_image_path  # noqa: E402

# ---------------------------------------------------------------- config --

SSR_ROOT = PROJECT_ROOT / "data" / "processed" / "faces_ssr"
REGIONS_ROOT = PROJECT_ROOT / "data" / "processed" / "regions"
REGIONS_INDEX = PROJECT_ROOT / "data" / "processed" / "regions_index.csv"
OUT_DIR = PROJECT_ROOT / "results" / "manual_qa"

# BGR tuples (cv2 convention) -- distinctness matters more than exact hue here.
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


def build_sheet(filename: str):
    ssr_path = SSR_ROOT / filename
    ssr_img = cv2.imread(str(ssr_path))
    if ssr_img is None:
        print(f"[skip] SSR image missing: {ssr_path}")
        return None

    orig_path, _, _ = resolve_image_path(filename)
    orig_img = cv2.imread(str(orig_path)) if orig_path else None
    if orig_img is None:
        print(f"[warn] original not found for {filename}, using blank panel")
        orig_img = np.zeros_like(ssr_img)

    basename = Path(filename).stem
    mst_folder = Path(filename).parent

    overlay = ssr_img.astype(np.float32).copy()
    found_any = False
    for region, color in REGION_COLORS_BGR.items():
        region_path = REGIONS_ROOT / mst_folder / f"{basename}_{region}.jpg"
        region_img = cv2.imread(str(region_path))
        if region_img is None:
            print(f"[warn] missing region file: {region_path}")
            continue
        mask = np.any(region_img > 0, axis=2)
        if mask.any():
            found_any = True
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array(color, dtype=np.float32)

    if not found_any:
        print(f"[warn] no non-empty region masks found for {filename}")

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    panels = [label(orig_img, "original"), label(ssr_img, "SSR"), label(overlay, "5 regions overlaid")]
    return np.hstack(panels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="random sample size")
    ap.add_argument("--filenames", nargs="*", default=None, help="specific filenames instead of a random sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not REGIONS_INDEX.is_file():
        raise SystemExit(f"ERROR: {REGIONS_INDEX} not found -- run segment_regions.py (7.3) first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_df = pd.read_csv(REGIONS_INDEX)
    all_filenames = index_df["filename"].unique().tolist()
    print(f"{REGIONS_INDEX.name}: {len(all_filenames)} unique images available")

    if args.filenames:
        chosen = args.filenames
    else:
        random.seed(args.seed)
        chosen = random.sample(all_filenames, min(args.n, len(all_filenames)))

    written = 0
    for filename in chosen:
        sheet = build_sheet(filename)
        if sheet is None:
            continue
        safe_name = Path(filename).stem.replace("/", "_") + "_qa.png"
        out_path = OUT_DIR / safe_name
        cv2.imwrite(str(out_path), sheet)
        print(f"wrote {out_path}")
        written += 1

    print(f"\n{written} / {len(chosen)} contact sheets written to {OUT_DIR}")
    print("\nWHAT TO LOOK FOR")
    print("  - do left_cheek / right_cheek land on the SUBJECT'S actual left/right,")
    print("    not mirrored or swapped?")
    print("  - on a frontal face, are left_cheek and right_cheek roughly symmetric")
    print("    in size and shape? (On a profile/turned face, asymmetry is expected")
    print("    and not a bug.)")
    print("  - does nose_bridge stay centered and narrow, without leaking into")
    print("    either cheek?")
    print("  - do any two regions visibly overlap? (Should be impossible -- 7.3")
    print("    already raises an error at import time if REGIONS indices clash --")
    print("    but confirm visually too.)")


if __name__ == "__main__":
    main()
