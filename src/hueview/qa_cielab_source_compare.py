"""
Phase 8.2 color-source comparison -- SSR-sourced vs original-sourced CIELAB.

The manuscript specifies computing CIELAB from "the per-region skin samples
produced by the SSR stage." But SSR's per-channel independent rescaling is
already known to distort hue severely (this is the exact mechanism that
caused Phase 7.4's hsv_source="ssr" default to fail on ~57% of images).
This script checks whether that same distortion has landed in the actual
CIELAB color features -- not just in a filtering threshold -- by computing
hue angle from CIELAB two ways for the same real pixels and checking what
fraction lands inside the manuscript's own stated skin band (0-50 deg,
with the same wrap-around near 360 deg used by hsv_skin_filter.py's relaxed
tier).

FIXED from the previous version: added multiprocessing (matching every
other full-dataset script in this pipeline) and a tqdm progress bar. The
previous version ran single-threaded with no progress indicator, which at
full dataset scale (40,449 images x 5 regions x 2 sources x an
skimage.color.rgb2lab call each) could run well over an hour with zero
visibility into progress.

Usage:
    python qa_cielab_source_compare.py --n 200
    python qa_cielab_source_compare.py --n 40449   # full dataset
"""

# BLAS thread-limiting MUST happen before numpy (or anything that imports
# numpy, including skimage and cv2) is imported anywhere in this process --
# these libraries read these env vars once, at their own internal BLAS
# thread-pool initialization. skimage.color.rgb2lab's matrix math (the
# RGB->XYZ linear transform) is BLAS-backed; BLAS spawns its own internal
# thread pool by default. With 12 worker processes each doing that
# repeatedly, you get dozens of threads competing for memory allocation at
# the OS level -- "OpenBLAS error: Memory allocation still failed after 10
# retries" is that oversubscription, not a real shortage of RAM. Forcing
# single-threaded BLAS per process is correct here because the parallelism
# already comes from the 12 separate OS processes, not from BLAS.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import random
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
from hueview.regions import decode_label_map, REGION_ORDER  # noqa: E402
from hueview.region_selector import RegionalConfigurationSelector, FULL_FACE  # noqa: E402
from hueview.cielab_features import build_cielab_vector  # noqa: E402

# ---------------------------------------------------------------- config --

INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
LABEL_MAPS_ROOT = PROJECT_ROOT / "data" / "processed" / "label_maps"
SSR_ROOT = PROJECT_ROOT / "data" / "processed" / "faces_ssr"

HUE_MIN, HUE_MAX = 0.0, 50.0
HUE_WRAP_MIN = 340.0  # same wrap-around used by hsv_skin_filter.py's relaxed tier


def in_skin_band(hue_deg: float) -> bool:
    return (HUE_MIN <= hue_deg <= HUE_MAX) or (hue_deg >= HUE_WRAP_MIN)


# NOTE: this script does NOT restore the true relaxed/geometry_fallback
# status from phase7_4_full_coverage_stats.csv, unlike run_phase7_5_batch.py.
# It doesn't need to: is_usable only depends on (status != EMPTY) and pixel
# count, and decode_label_map()'s own OK/EMPTY reconstruction already gets
# that exactly right on its own -- the finer status label this script would
# be restoring is never read or reported here. Skipping it also avoids 12
# worker processes each independently parsing a 202k-row CSV at startup,
# which is what caused the earlier out-of-memory crash.

# --------------------------------------------------------------- worker --


def process_one(filename: str):
    label_path = LABEL_MAPS_ROOT / Path(filename).with_suffix(".png")
    ssr_path = SSR_ROOT / filename

    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    ssr_bgr = cv2.imread(str(ssr_path))
    if label is None or ssr_bgr is None:
        return None

    ssr_rgb = cv2.cvtColor(ssr_bgr, cv2.COLOR_BGR2RGB)
    patches = decode_label_map(label, ssr_rgb)

    orig_path, _, _ = resolve_image_path(filename)
    if orig_path is None:
        return None
    orig_bgr = cv2.imread(str(orig_path))
    if orig_bgr is None:
        return None
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    selector = RegionalConfigurationSelector()
    rows = []
    for cfg in selector.route(patches, image_id=filename):
        if cfg.config == FULL_FACE:
            continue  # per-region only, to avoid double counting

        vec_ssr = build_cielab_vector(cfg)
        vec_orig = build_cielab_vector(cfg, original_rgb=orig_rgb)

        for label_name, vec in (("ssr", vec_ssr), ("original", vec_orig)):
            L, a, b = vec
            hue = float(np.degrees(np.arctan2(b, a)) % 360)
            rows.append({
                "filename": filename,
                "region": cfg.config,
                "source": label_name,
                "L": L, "a": a, "b": b,
                "hue": hue,
                "in_band": in_skin_band(hue),
            })
    return rows


# --------------------------------------------------------------- main --


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not INDEX_PATH.is_file():
        raise SystemExit(f"ERROR: {INDEX_PATH} not found.")

    index_df = pd.read_csv(INDEX_PATH)
    random.seed(args.seed)
    chosen = random.sample(index_df["filename"].tolist(), min(args.n, len(index_df)))
    print(f"Sampling {len(chosen)} images, {cpu_count()} workers")

    all_rows = []
    with Pool(processes=cpu_count()) as pool:
        for rows in tqdm(pool.imap_unordered(process_one, chosen), total=len(chosen)):
            if rows:
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[error] nothing processed")
        return

    print(f"\nProcessed {len(chosen)} images, {len(df) // 2} region observations per source\n")
    print("FRACTION OF HUE ANGLES LANDING IN THE SKIN BAND (0-50 deg, wrap >=340 deg)")
    print("-" * 74)
    pivot = df.pivot_table(index="region", columns="source", values="in_band", aggfunc="mean")
    pivot = pivot.reindex(REGION_ORDER)
    print(pivot.round(3).to_string())

    print("\nOVERALL (all regions pooled)")
    print("-" * 74)
    overall = df.groupby("source")["in_band"].mean()
    print(overall.round(3).to_string())

    print("\nMEAN a*, b* BY SOURCE (should both be positive for real skin)")
    print("-" * 74)
    print(df.groupby("source")[["a", "b"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
