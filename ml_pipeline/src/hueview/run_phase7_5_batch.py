"""
Phase 7.5 -- Full-Dataset Regional Configuration Routing (batch run)

RegionalConfigurationSelector is a generator meant to be called once per
image inside the Phase 8 training loop, per its own docstring -- it holds
no learned parameters and produces no features. This script exists to
produce the artifact its docstring calls for regardless:
results/phase7_routing_log.csv, "the evidence for two things the panel is
likely to ask about: how many regions needed the relaxed or fallback
filter, and how many images contributed fewer than five regions to their
Full Face vector" -- across the REAL full dataset, not a QA sample.

Masks and per-region images are reconstructed from the compact label maps
written by run_phase7_4_batch.py via regions.decode_label_map(), using the
already-generated faces_ssr/ images (converted to RGB, matching the
convention the patches were originally built with).

ACCURACY NOTE: decode_label_map() can only reconstruct STATUS_OK or
STATUS_EMPTY per region -- the label map has no bits for "relaxed" vs
"geometry_fallback" (see its own docstring: "If you need exact status
replay, join against the routing log on filename"). This script does
exactly that: it joins phase7_4_full_coverage_stats.csv (the real per-
region status from the actual HSV filtering run) onto each reconstructed
patch before routing, so the log's worst-status-per-configuration field is
accurate rather than collapsed to OK/EMPTY. Usability (skip_unusable,
is_usable) is unaffected either way -- that only ever depended on pixel
counts, not the status label.

Input:
    data/processed/landmarks_index.csv                -- filename universe
    data/processed/label_maps/<filename, .png>        -- from 7.4
    data/processed/faces_ssr/<filename>                -- from 7.2
    data/processed/phase7_4_full_coverage_stats.csv    -- real per-region
                                                           status, from 7.4

Output:
    results/phase7_routing_log.csv -- one row per image, from
                                       RegionalConfigurationSelector.routing_record()
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

from ml_pipeline.src.hueview.regions import decode_label_map, REGION_ORDER  # noqa: E402
from ml_pipeline.src.hueview.region_selector import RegionalConfigurationSelector  # noqa: E402

# ---------------------------------------------------------------- config --

INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
LABEL_MAPS_ROOT = PROJECT_ROOT / "data" / "processed" / "label_maps"
SSR_ROOT = PROJECT_ROOT / "data" / "processed" / "faces_ssr"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "phase7_4_full_coverage_stats.csv"
OUT_PATH = PROJECT_ROOT / "results" / "phase7_routing_log.csv"
FAILURES_PATH = PROJECT_ROOT / "data" / "processed" / "phase7_5_failures.csv"

LIMIT = None  # e.g. 50 to sanity-check before the full run


def load_status_lookup() -> dict:
    """{(filename, region): true_status} from the real Phase 7.4 run --
    loaded once per process (main + each worker), not once per job."""
    df = pd.read_csv(STATS_PATH)
    return {(row.filename, row.region): row.status for row in df.itertuples()}


STATUS_LOOKUP = load_status_lookup() if STATS_PATH.is_file() else {}

# --------------------------------------------------------------- worker --


def process_one(filename: str):
    label_path = LABEL_MAPS_ROOT / Path(filename).with_suffix(".png")
    ssr_path = SSR_ROOT / filename

    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    ssr_bgr = cv2.imread(str(ssr_path))
    if label is None or ssr_bgr is None:
        return (filename, None, "label_map_or_ssr_missing")

    ssr_rgb = cv2.cvtColor(ssr_bgr, cv2.COLOR_BGR2RGB)
    patches = decode_label_map(label, ssr_rgb)

    # Restore the true status -- decode_label_map only knows ok/empty.
    for region, patch in patches.items():
        true_status = STATUS_LOOKUP.get((filename, region))
        if true_status is not None:
            patch.status = true_status

    selector = RegionalConfigurationSelector()
    outputs = list(selector.route(patches, image_id=filename))
    row = selector.routing_record(filename, outputs)
    return (filename, row, None)


# --------------------------------------------------------------- main --


def run():
    print(f"Project root: {PROJECT_ROOT}")

    if not INDEX_PATH.is_file():
        raise SystemExit(f"ERROR: {INDEX_PATH} not found -- run landmark_extraction.py (7.1) first.")
    if not STATS_PATH.is_file():
        raise SystemExit(f"ERROR: {STATS_PATH} not found -- run run_phase7_4_batch.py first.")

    index_df = pd.read_csv(INDEX_PATH)
    filenames = index_df["filename"].tolist()
    print(f"{INDEX_PATH.name}: {len(filenames)} images")

    if LIMIT:
        filenames = filenames[:LIMIT]
        print(f"LIMIT set -- only processing {len(filenames)} images")

    rows = []
    failures = []
    with Pool(processes=cpu_count()) as pool:
        for filename, row, error in tqdm(pool.imap_unordered(process_one, filenames), total=len(filenames)):
            if error is not None:
                failures.append({"filename": filename, "reason": error})
            else:
                rows.append(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nProcessed {len(rows)} / {len(filenames)} images successfully")
    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
        print(f"Failed: {len(failures)} (logged to {FAILURES_PATH})")
    else:
        print("Failed: 0")
    print(f"Saved routing log to {OUT_PATH}")

    if rows:
        print("\nFULL FACE COMPLETENESS")
        print("-" * 60)
        for k in range(5, -1, -1):
            n = int((df["full_face_regions_present"] == k).sum())
            if n:
                tag = "all 5 regions present" if k == 5 else f"exactly {k} region(s) present"
                print(f"  {n} images with {tag}")

        print("\nCONFIG STATUS COUNTS")
        print("-" * 60)
        for config in list(REGION_ORDER) + ["full_face"]:
            col = f"{config}_status"
            if col in df.columns:
                print(f"{config:12s}: {df[col].value_counts().to_dict()}")


if __name__ == "__main__":
    run()
