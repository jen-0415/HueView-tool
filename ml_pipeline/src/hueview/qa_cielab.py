"""
Manual QA for Phase 8.2 -- CIELAB feature extraction.

Reuses the same decode_label_map() + status-restore pattern as
run_phase7_5_batch.py to reconstruct real RegionPatch objects, routes them
through the real RegionalConfigurationSelector, and builds real CIELAB
vectors via cielab_features.build_cielab_vector() -- on your actual data,
not synthetic pixels.

What to check in the output:
    - L* should sit roughly in the 20-85 range for real skin, varying
      with the subject's tone (lower = darker skin).
    - a* should be positive (reddish) for essentially all real skin.
    - b* should be positive (yellowish) for essentially all real skin.
    - hue_angle (atan2(b*, a*), degrees) should mostly fall in the
      manuscript's stated skin band, roughly 0-50 deg for warm tones with
      some spread -- this is the same quantity Phase 9's undertone
      descriptor will use, so if it looks wrong here, Phase 9 will inherit
      the bug.
    - full_face's cielab_dim should always be 15; every single-region
      config should always be 3.

Usage:
    python qa_cielab.py --n 10
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

from ml_pipeline.src.hueview.regions import decode_label_map  # noqa: E402
from ml_pipeline.src.hueview.region_selector import RegionalConfigurationSelector  # noqa: E402
from ml_pipeline.src.hueview.cielab_features import build_cielab_vector  # noqa: E402

# ---------------------------------------------------------------- config --

INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
LABEL_MAPS_ROOT = PROJECT_ROOT / "data" / "processed" / "label_maps"
SSR_ROOT = PROJECT_ROOT / "data" / "processed" / "faces_ssr"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "phase7_4_full_coverage_stats.csv"


def load_status_lookup() -> dict:
    df = pd.read_csv(STATS_PATH)
    return {(row.filename, row.region): row.status for row in df.itertuples()}


def load_patches(filename: str):
    label_path = LABEL_MAPS_ROOT / Path(filename).with_suffix(".png")
    ssr_path = SSR_ROOT / filename

    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    ssr_bgr = cv2.imread(str(ssr_path))
    if label is None or ssr_bgr is None:
        return None

    ssr_rgb = cv2.cvtColor(ssr_bgr, cv2.COLOR_BGR2RGB)
    return decode_label_map(label, ssr_rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--filenames", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for required in (INDEX_PATH, STATS_PATH):
        if not required.is_file():
            raise SystemExit(f"ERROR: {required} not found.")

    index_df = pd.read_csv(INDEX_PATH)
    status_lookup = load_status_lookup()

    if args.filenames:
        chosen = args.filenames
    else:
        random.seed(args.seed)
        chosen = random.sample(index_df["filename"].tolist(), min(args.n, len(index_df)))

    selector = RegionalConfigurationSelector()

    for filename in chosen:
        patches = load_patches(filename)
        if patches is None:
            print(f"[skip] {filename}: label map or SSR image missing")
            continue

        for region, patch in patches.items():
            true_status = status_lookup.get((filename, region))
            if true_status is not None:
                patch.status = true_status

        print(f"\n{filename}")
        print("-" * 70)
        for cfg in selector.route(patches, image_id=filename):
            vec = build_cielab_vector(cfg)
            if cfg.cielab_dim == 3:
                L, a, b = vec
                hue = np.degrees(np.arctan2(b, a)) % 360
                print(f"  {cfg.config:12s} L*={L:6.2f} a*={a:6.2f} b*={b:6.2f} "
                      f"hue={hue:6.1f} deg  status={cfg.status:18s} "
                      f"px={cfg.n_valid_px}")
            else:
                print(f"  {cfg.config:12s} dim={vec.shape[0]:2d} "
                      f"(expected 15) status={cfg.status:18s} "
                      f"imputed={cfg.imputed_regions or 'none'} "
                      f"px={cfg.n_valid_px}")
                per_region = vec.reshape(5, 3)
                for name, (L, a, b) in zip(cfg.component_order(), per_region):
                    hue = np.degrees(np.arctan2(b, a)) % 360
                    print(f"      {name:12s} L*={L:6.2f} a*={a:6.2f} b*={b:6.2f} hue={hue:6.1f} deg")


if __name__ == "__main__":
    main()
