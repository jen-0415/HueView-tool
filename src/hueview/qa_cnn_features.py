"""
Manual QA for Phase 8.1 -- CNN feature extraction.

Same decode_label_map() + RegionalConfigurationSelector pipeline already
used for the Phase 8.2 CIELAB QA, now feeding cfg.image through the shared
EfficientNetB0 backbone. Checks output shape and basic sanity (no NaN/Inf,
non-degenerate variance) rather than semantic correctness -- CNN features
aren't human-interpretable the way CIELAB values are, so there's no
"plausible range" to eyeball here the way there was for L*/a*/b*.

First run downloads ImageNet-pretrained weights (~29 MB) if not already
cached -- needs internet access once.

Runs on CPU (no native Windows GPU support for TensorFlow since v2.11, per
this project's established setup) -- fine for a small QA sample, but this
is why full-dataset feature extraction and training happen on Colab.

Usage:
    python qa_cnn_features.py --n 5
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

from hueview.regions import decode_label_map  # noqa: E402
from hueview.region_selector import RegionalConfigurationSelector  # noqa: E402
from hueview.cnn_features import build_cnn_backbone, extract_cnn_features, CNN_OUTPUT_DIM  # noqa: E402

# ---------------------------------------------------------------- config --

INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
LABEL_MAPS_ROOT = PROJECT_ROOT / "data" / "processed" / "label_maps"
SSR_ROOT = PROJECT_ROOT / "data" / "processed" / "faces_ssr"


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
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--filenames", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not INDEX_PATH.is_file():
        raise SystemExit(f"ERROR: {INDEX_PATH} not found.")

    index_df = pd.read_csv(INDEX_PATH)
    if args.filenames:
        chosen = args.filenames
    else:
        random.seed(args.seed)
        chosen = random.sample(index_df["filename"].tolist(), min(args.n, len(index_df)))

    print("Building shared EfficientNetB0 backbone (first run downloads ImageNet weights)...")
    model = build_cnn_backbone()
    print(f"Backbone ready. Output dim: {CNN_OUTPUT_DIM}\n")

    selector = RegionalConfigurationSelector()

    for filename in chosen:
        patches = load_patches(filename)
        if patches is None:
            print(f"[skip] {filename}: label map or SSR image missing")
            continue

        print(filename)
        print("-" * 74)
        for cfg in selector.route(patches, image_id=filename):
            vec = extract_cnn_features(model, cfg.image)
            ok_shape = vec.shape == (CNN_OUTPUT_DIM,)
            has_nan = bool(np.isnan(vec).any())
            has_inf = bool(np.isinf(vec).any())
            print(
                f"  {cfg.config:12s} shape={vec.shape} ok_shape={ok_shape} "
                f"nan={has_nan} inf={has_inf} mean={vec.mean():.4f} "
                f"std={vec.std():.4f} status={cfg.status}"
            )
        print()


if __name__ == "__main__":
    main()
