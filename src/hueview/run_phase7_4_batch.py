"""
Phase 7.4 -- Full-Dataset HSV Skin Filtering (batch run)

Runs the frozen Phase 7.4 methodology (configs/hsv_skin_thresholds.json,
hsv_source="original" -- chosen over the manuscript-literal "ssr" reading
based on the n=300 comparison: ~57% geometry_fallback under "ssr" vs.
single digits under "original", across every region) across the full
landmarked dataset.

Follows regions.py's intended caching contract rather than 7.3's
separate-JPEG-per-region style: each image's 5 geometric masks and their
post-HSV skin masks are packed via encode_label_map() into ONE compact PNG
per image. SSR itself is NOT re-saved here -- per regions.py's own
docstring, it's cheap enough to recompute on the fly from faces_ssr/ at
Phase 8 read time (via decode_label_map(label, ssr_image)); only the masks,
which are expensive to recompute (MediaPipe + polygon + HSV + morphology),
are cached.

This does NOT replace data/processed/regions/ (7.3's crops) -- those stay
as-is, already used for visual QA. This is a separate, Phase-8-ready
artifact.

Input:
    data/processed/landmarks.npy        -- (N, 468, 3), from 7.1
    data/processed/landmarks_index.csv  -- row_index, filename
    data/processed/faces_ssr/<filename> -- SSR output, from 7.2
    configs/hsv_skin_thresholds.json    -- frozen Phase 7.4 config
    (each filename's original pre-SSR crop is located via
     path_resolver.resolve_image_path -- needed because the frozen config
     uses hsv_source="original")

Output:
    data/processed/label_maps/<filename, .png>       -- packed masks,
                                                         one file per image
    data/processed/phase7_4_full_coverage_stats.csv  -- filename, region,
                                                         status, retention --
                                                         the real,
                                                         full-dataset
                                                         Sampling Data
                                                         evidence, not the
                                                         n=300 QA sample
    data/processed/phase7_4_failures.csv             -- filename, reason
                                                         (only if any)
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
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline.path_resolver import resolve_image_path  # noqa: E402
from hueview.segment_regions import REGIONS, build_region_mask  # noqa: E402
from hueview.hsv_skin_filter import FilterConfig, filter_all_regions, summarize  # noqa: E402
from hueview.regions import encode_label_map  # noqa: E402

# ---------------------------------------------------------------- config --

LANDMARKS_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
SSR_ROOT = PROJECT_ROOT / "data" / "processed" / "faces_ssr"
LABEL_MAPS_ROOT = PROJECT_ROOT / "data" / "processed" / "label_maps"
CONFIG_PATH = PROJECT_ROOT / "configs" / "hsv_skin_thresholds.json"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "phase7_4_full_coverage_stats.csv"
FAILURES_PATH = PROJECT_ROOT / "data" / "processed" / "phase7_4_failures.csv"

LIMIT = None  # e.g. 50 to sanity-check before the full run

CONFIG = FilterConfig.from_json(CONFIG_PATH)

# --------------------------------------------------------------- worker --


def process_one(args):
    filename, row_index, landmarks_px = args

    ssr_path = SSR_ROOT / filename
    ssr_bgr = cv2.imread(str(ssr_path))
    if ssr_bgr is None:
        return (filename, None, "ssr_image_missing_or_unreadable")

    # hsv_skin_filter.py requires RGB -- faces_ssr/ files are BGR on disk.
    ssr_rgb = cv2.cvtColor(ssr_bgr, cv2.COLOR_BGR2RGB)

    geometric_masks = {
        name: build_region_mask(landmarks_px, idx, ssr_rgb.shape).astype(bool)
        for name, idx in REGIONS.items()
    }

    reference_rgb = None
    if CONFIG.hsv_source == "original":
        orig_path, _, _ = resolve_image_path(filename)
        if orig_path is None:
            return (filename, None, "original_not_found")
        orig_bgr = cv2.imread(str(orig_path))
        if orig_bgr is None:
            return (filename, None, "original_unreadable")
        reference_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    patches = filter_all_regions(
        ssr_image=ssr_rgb,
        geometric_masks=geometric_masks,
        config=CONFIG,
        reference_image=reference_rgb,
    )

    label_map = encode_label_map(patches, ssr_rgb.shape[:2])
    out_path = LABEL_MAPS_ROOT / Path(filename).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), label_map)
    if not ok:
        return (filename, None, "label_map_write_failed")

    stats = summarize(patches)
    rows = [{"filename": filename, "region": region, **s} for region, s in stats.items()]
    return (filename, rows, None)


# --------------------------------------------------------------- main --


def run():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using frozen config: {CONFIG_PATH} (hsv_source={CONFIG.hsv_source})")

    if not LANDMARKS_PATH.is_file() or not INDEX_PATH.is_file():
        raise SystemExit("ERROR: landmarks.npy / landmarks_index.csv not found -- run landmark_extraction.py (7.1) first.")
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"ERROR: {CONFIG_PATH} not found -- freeze the Phase 7.4 config first.")

    landmarks_array = np.load(LANDMARKS_PATH)
    index_df = pd.read_csv(INDEX_PATH)
    print(f"Loaded {len(index_df)} landmark rows, landmarks array shape {landmarks_array.shape}")

    jobs = [
        (row["filename"], row["row_index"], landmarks_array[row["row_index"]])
        for _, row in index_df.iterrows()
    ]

    if LIMIT:
        jobs = jobs[:LIMIT]
        print(f"LIMIT set -- only processing {len(jobs)} images")

    print(f"Total: {len(jobs)} images, {cpu_count()} workers")

    all_rows = []
    failures = []

    with Pool(processes=cpu_count()) as pool:
        for filename, rows, error in tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs)):
            if error is not None:
                failures.append({"filename": filename, "reason": error})
            else:
                all_rows.extend(rows)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(STATS_PATH, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)

    n_success = len(jobs) - len(failures)
    print(f"\nProcessed {n_success} / {len(jobs)} images successfully")
    print(f"Failed: {len(failures)} (logged to {FAILURES_PATH.name})" if failures else "Failed: 0")
    print(f"Saved label maps to {LABEL_MAPS_ROOT}")
    if all_rows:
        print(f"Saved coverage stats to {STATS_PATH}")
        stats_df = pd.DataFrame(all_rows)
        print("\nRETENTION BY REGION (this run)")
        print(stats_df.pivot_table(index="region", values="retention", aggfunc="mean").round(3).to_string())
        print("\nFILTER TIER USAGE (this run)")
        print(stats_df.groupby(["region", "status"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    run()
