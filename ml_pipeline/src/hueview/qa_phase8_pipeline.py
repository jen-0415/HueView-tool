"""
End-to-end Phase 8 QA -- decode -> route -> CNN (8.1) -> CIELAB (8.2) ->
fusion + classification (8.3), all in one pass, on real data.

The heads are freshly initialized (not trained), so the softmax output
here is meaningless as a prediction -- this script checks PLUMBING, not
classification quality: does every configuration produce a valid (6,)
probability distribution that sums to 1, does assert_feature_shapes() pass
for every configuration (3-dim CIELAB routed to region_head, 15-dim routed
to full_face_head), and does nothing NaN/Inf/crash along the way.

Usage:
    python qa_phase8_pipeline.py --n 3
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
from ml_pipeline.src.hueview.regions import decode_label_map  # noqa: E402
from ml_pipeline.src.hueview.region_selector import RegionalConfigurationSelector, FULL_FACE  # noqa: E402
from ml_pipeline.src.hueview.cielab_features import build_cielab_vector  # noqa: E402
from ml_pipeline.src.hueview.cnn_features import build_cnn_backbone, extract_cnn_features  # noqa: E402
from ml_pipeline.src.hueview.hybrid_classifier import build_hueview_heads, predict_configuration, REGION_HEAD, FULL_FACE_HEAD  # noqa: E402

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
    ap.add_argument("--n", type=int, default=3)
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

    print("Building shared EfficientNetB0 backbone...")
    cnn_model = build_cnn_backbone()
    print("Building fusion/classification heads (freshly initialized, untrained --")
    print("softmax output below is plumbing-only, not a real prediction)...")
    heads = build_hueview_heads()
    print("Ready.\n")

    selector = RegionalConfigurationSelector()
    n_checked = 0
    n_ok = 0

    for filename in chosen:
        patches = load_patches(filename)
        if patches is None:
            print(f"[skip] {filename}: label map or SSR image missing")
            continue

        orig_path, _, _ = resolve_image_path(filename)
        orig_rgb = None
        if orig_path:
            orig_bgr = cv2.imread(str(orig_path))
            if orig_bgr is not None:
                orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        print(filename)
        print("-" * 78)
        for cfg in selector.route(patches, image_id=filename):
            n_checked += 1
            try:
                cnn_vec = extract_cnn_features(cnn_model, cfg.image)
                cielab_vec = build_cielab_vector(cfg, original_rgb=orig_rgb)
                probs = predict_configuration(cfg, cnn_vec, cielab_vec, heads)

                head_used = FULL_FACE_HEAD if cfg.config == FULL_FACE else REGION_HEAD
                sums_to_one = bool(np.isclose(probs.sum(), 1.0, atol=1e-4))
                shape_ok = probs.shape == (6,)
                clean = not (np.isnan(probs).any() or np.isinf(probs).any())

                ok = sums_to_one and shape_ok and clean
                n_ok += int(ok)

                print(
                    f"  {cfg.config:12s} head={head_used:14s} cielab_dim={cfg.cielab_dim:2d} "
                    f"probs_shape={probs.shape} sums_to_1={sums_to_one} clean={clean} "
                    f"argmax=SCC-{probs.argmax() + 1}"
                )
            except Exception as e:
                print(f"  {cfg.config:12s} FAILED: {e!r}")
        print()

    print(f"Checked {n_checked} configuration outputs, {n_ok} passed all plumbing checks.")


if __name__ == "__main__":
    main()
