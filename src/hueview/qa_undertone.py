"""
Manual QA for Phase 9 -- undertone descriptor.

Same decode_label_map() + RegionalConfigurationSelector + CIELAB pipeline
already used for Phase 8.2, now computing hue angle and Warm/Neutral/Cool
per region, plus the Full Face majority vote (with the build-plan's
tie-break, since the manuscript doesn't specify one).

No ground truth exists to check this against (STW has no undertone labels,
per the manuscript) -- this checks plumbing and internal consistency: does
every hue/label pair respect the stated thresholds, does the majority
label match a manual tally over the printed per-region labels, and how
often does a genuine tie actually occur across a real sample (worth
knowing before treating the tie-break rule as a rare edge case).

Usage:
    python qa_undertone.py --n 15
"""

import argparse
import random
import sys
from collections import Counter
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
from hueview.regions import decode_label_map  # noqa: E402
from hueview.region_selector import RegionalConfigurationSelector, FULL_FACE  # noqa: E402
from hueview.cielab_features import build_cielab_vector  # noqa: E402
from hueview.undertone import compute_undertone_descriptor  # noqa: E402

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
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not INDEX_PATH.is_file():
        raise SystemExit(f"ERROR: {INDEX_PATH} not found.")

    index_df = pd.read_csv(INDEX_PATH)
    random.seed(args.seed)
    chosen = random.sample(index_df["filename"].tolist(), min(args.n, len(index_df)))

    selector = RegionalConfigurationSelector()
    n_full_face = 0
    n_tied = 0
    n_mismatch = 0

    for filename in chosen:
        patches = load_patches(filename)
        if patches is None:
            print(f"[skip] {filename}")
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
            cielab_vec = build_cielab_vector(cfg, original_rgb=orig_rgb)
            result = compute_undertone_descriptor(cfg, cielab_vec)

            if cfg.config == FULL_FACE:
                n_full_face += 1
                # Manual tally, independent of majority_undertone(), as a
                # cross-check that the vote logic did what it claims.
                labels = [r["undertone"] for r in result["regions"].values()]
                tally = Counter(labels)
                manual_top = tally.most_common()
                manual_leaders = [lbl for lbl, c in manual_top if c == manual_top[0][1]]
                manually_tied = len(manual_leaders) > 1

                if result["was_tied"]:
                    n_tied += 1
                if manually_tied != result["was_tied"]:
                    n_mismatch += 1
                    print(f"  ** MISMATCH: tally={dict(tally)} but was_tied={result['was_tied']} **")

                region_str = ", ".join(
                    f"{n}={r['undertone']}({r['hue_deg']:.0f} deg)" for n, r in result["regions"].items()
                )
                print(
                    f"  {cfg.config:12s} tally={dict(tally)!s:35} "
                    f"majority={result['majority_undertone']:8s} tied={result['was_tied']}"
                )
                print(f"      {region_str}")
            else:
                only = list(result["regions"].values())[0]
                print(f"  {cfg.config:12s} hue={only['hue_deg']:6.1f} deg  undertone={only['undertone']}")
        print()

    print(f"Full Face configs checked: {n_full_face}")
    print(f"Ties encountered: {n_tied} ({100 * n_tied / max(n_full_face, 1):.1f}%)")
    print(f"Manual-tally mismatches: {n_mismatch} (should be 0)")


if __name__ == "__main__":
    main()
