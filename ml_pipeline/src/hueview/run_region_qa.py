"""
Phase 7 Checkpoint — Visual QA and Coverage Statistics for 7.4 / 7.5
====================================================================

The build plan's Phase 7 checkpoint says:

    "For a sample of test images, visually inspect the 5 regional masks
    overlaid on the face — confirm they land where anatomically expected,
    and that HSV filtering isn't over- or under-stripping skin pixels. Do
    this before any CNN training; segmentation bugs are expensive to catch
    later."

This script is that checkpoint. It runs a sample of images through 7.4 and
7.5 and writes:

  1. results/phase7_qa/<filename>.png — a contact sheet per image:
        original | SSR | polygon masks | post-HSV skin masks
  2. results/phase7_coverage_stats.csv — per-image, per-region retention and
     status. These are the numbers for your Sampling Data write-up and for
     answering "how often did the filter fall back?" in the defense.
  3. results/phase7_routing_log.csv — the Phase 7.5 routing record.

It also has a `--compare-hsv-source` mode that runs both the "ssr" and
"original" decision paths on the same sample and prints retention side by
side, so the choice between them is made on evidence rather than preference.
See the SSR interaction note in `hsv_skin_filter.py`.


RUNNING BEFORE 7.1-7.3 EXIST

Phase 7.4 needs geometric masks from 7.3, which needs landmarks from 7.1 and
an SSR image from 7.2. So that 7.4 and 7.5 can be tested and reviewed
immediately, this script falls back to two reference implementations when
the real modules are not importable:

  * SSR: a direct implementation of the manuscript's formula,
    R(x,y) = log(I(x,y)) - log(F(x,y) * I(x,y)), sigma = 80.
  * Region masks: rectangles built from Table 3's stated coverage areas
    (top 25% height / center 60% width for the forehead, and so on).

The fallback masks are RECTANGLES, not landmark polygons. They are correct
enough to validate that 7.4 and 7.5 behave, and wrong enough that you must
not train on them. Once `src/hueview/segmentation.py` exposes
`segment_regions(image, landmarks) -> {region: bool mask}`, this script
picks it up automatically and the fallback stops being used. The banner at
the top of the run tells you which path was taken.


USAGE

    # from the repo root
    python -m src.hueview.run_region_qa --n 12
    python -m src.hueview.run_region_qa --n 40 --compare-hsv-source
    python -m src.hueview.run_region_qa --manifest data/processed/test.csv --n 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# Allow both `python -m src.hueview.run_region_qa` and direct execution.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ml_pipeline.src.hueview.hsv_skin_filter import (  # type: ignore
        DEFAULT_CONFIG, FilterConfig, filter_all_regions, summarize,
    )
    from ml_pipeline.src.hueview.region_selector import RegionalConfigurationSelector  # type: ignore
    from ml_pipeline.src.hueview.regions import (  # type: ignore
        REGION_ORDER, TABLE3_COVERAGE, STATUS_OK,
    )
else:
    from .hsv_skin_filter import DEFAULT_CONFIG, FilterConfig, filter_all_regions, summarize
    from .region_selector import RegionalConfigurationSelector
    from .regions import REGION_ORDER, TABLE3_COVERAGE, STATUS_OK


# --------------------------------------------------------------------------
# Reference implementations (used only when 7.1-7.3 are not importable)
# --------------------------------------------------------------------------


def reference_ssr(image_rgb: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """
    Single-Scale Retinex, per the manuscript's Stage 2 formula.

        R(x, y) = log(I(x, y)) - log(F(x, y) * I(x, y))

    applied independently to R, G and B, then rescaled to 0-255.

    This exists so the QA harness runs standalone. Your Phase 7.2 module is
    the authoritative implementation — if it differs from this in any way
    (epsilon handling, rescaling strategy), 7.2 wins and this should be
    deleted rather than reconciled.
    """
    img = image_rgb.astype(np.float32) + 1.0  # +1 avoids log(0)
    out = np.zeros_like(img)

    for c in range(3):
        channel = img[:, :, c]
        # cv2 picks the kernel size from sigma; at sigma=80 on a 224x224
        # image the surround is effectively the whole face, which is the
        # intent — SSR is estimating the global illumination field.
        blurred = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
        retinex = np.log(channel) - np.log(blurred + 1.0)
        lo, hi = retinex.min(), retinex.max()
        out[:, :, c] = 0.0 if hi - lo < 1e-8 else (retinex - lo) / (hi - lo) * 255.0

    return np.clip(out, 0, 255).astype(np.uint8)


def reference_region_masks(shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Rectangular region masks from Table 3's coverage areas.

    NOT a substitute for landmark-derived polygons. See the module docstring.
    """
    h, w = shape
    masks: Dict[str, np.ndarray] = {}
    claimed = np.zeros((h, w), dtype=bool)

    for name in REGION_ORDER:
        y0f, y1f, x0f, x1f = TABLE3_COVERAGE[name]
        y0, y1 = int(y0f * h), int(y1f * h)
        x0, x1 = int(x0f * w), int(x1f * w)
        m = np.zeros((h, w), dtype=bool)
        m[y0:y1, x0:x1] = True
        # Table 3's rectangles overlap (nose bridge sits inside the cheek
        # band). Regions must be disjoint or the label-map codec loses
        # pixels, so earlier regions in REGION_ORDER keep contested pixels.
        m &= ~claimed
        claimed |= m
        masks[name] = m

    return masks


def _try_real_pipeline():
    """Import the real 7.2/7.3 modules if they exist yet."""
    ssr_fn = None
    seg_fn = None
    try:
        from .ssr import apply_ssr  # type: ignore
        ssr_fn = apply_ssr
    except Exception:
        pass
    try:
        from .segmentation import segment_regions  # type: ignore
        seg_fn = segment_regions
    except Exception:
        pass
    return ssr_fn, seg_fn


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

_REGION_COLORS = {
    "forehead":    (255, 99, 71),
    "left_cheek":  (60, 179, 113),
    "right_cheek": (65, 105, 225),
    "nose_bridge": (255, 215, 0),
    "jawline":     (186, 85, 211),
}


def _overlay(base_rgb: np.ndarray, masks: Dict[str, np.ndarray], alpha: float = 0.45) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    for name, mask in masks.items():
        if mask is None or not mask.any():
            continue
        color = np.array(_REGION_COLORS.get(name, (255, 255, 255)), dtype=np.float32)
        out[mask] = (1 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a caption strip under a panel."""
    h, w = img.shape[:2]
    strip = np.full((22, w, 3), 245, dtype=np.uint8)
    cv2.putText(strip, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([img, strip])


def build_contact_sheet(
    original: np.ndarray,
    ssr: np.ndarray,
    geometric: Dict[str, np.ndarray],
    skin: Dict[str, np.ndarray],
    stats: Dict[str, Dict],
) -> np.ndarray:
    """Four panels side by side, saved as BGR for cv2.imwrite."""
    retention = np.mean([s["retention"] for s in stats.values()]) if stats else 0.0
    n_escalated = sum(1 for s in stats.values() if s["status"] != STATUS_OK)

    panels = [
        _label(original, "1. original crop"),
        _label(ssr, "2. SSR (sigma=80)"),
        _label(_overlay(ssr, geometric), "3. 7.3 polygons"),
        _label(_overlay(ssr, skin), f"4. 7.4 skin  ret={retention:.0%} esc={n_escalated}"),
    ]
    sheet = np.hstack(panels)
    return cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def load_image(path: Path, size: int = 224) -> Optional[np.ndarray]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    if bgr.shape[0] != size or bgr.shape[1] != size:
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resolve_paths(manifest: Path, images_dir: Path, n: int, seed: int) -> list:
    """
    Pick a sample of image paths.

    Prefers the frozen manifest so the sample is reproducible and traceable
    to a split. Falls back to globbing the images directory if the manifest
    is missing a resolved-path column, which keeps the checkpoint runnable
    on a partially-downloaded dataset.
    """
    if manifest.exists():
        df = pd.read_csv(manifest)
        for col in ("resolved_path", "path", "filepath", "filename"):
            if col in df.columns:
                sample = df.sample(min(n, len(df)), random_state=seed)
                paths = []
                for value in sample[col].astype(str):
                    p = Path(value)
                    if not p.is_absolute():
                        p = images_dir / p.name
                    if p.exists():
                        paths.append(p)
                if paths:
                    return paths
        print(f"[warn] {manifest} had no usable path column; falling back to glob")

    files = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rng = np.random.default_rng(seed)
    if len(files) > n:
        idx = rng.choice(len(files), size=n, replace=False)
        files = [files[i] for i in sorted(idx)]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 7.4 / 7.5 visual QA and coverage stats")
    ap.add_argument("--manifest", default="data/processed/test.csv")
    ap.add_argument("--images-dir", default="data/processed/images")
    ap.add_argument("--out-dir", default="results/phase7_qa")
    ap.add_argument("--n", type=int, default=12, help="sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hsv-source", choices=["ssr", "original"], default="ssr")
    ap.add_argument(
        "--compare-hsv-source",
        action="store_true",
        help="run both decision paths and print retention side by side",
    )
    ap.add_argument("--no-sheets", action="store_true", help="stats only, skip contact sheets")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ssr_fn, seg_fn = _try_real_pipeline()
    print("=" * 74)
    print("Phase 7 checkpoint — HSV skin filtering (7.4) + configuration routing (7.5)")
    print(f"  SSR source          : {'src/hueview/ssr.py (7.2)' if ssr_fn else 'REFERENCE fallback'}")
    print(f"  Region mask source  : {'src/hueview/segmentation.py (7.3)' if seg_fn else 'REFERENCE fallback — Table 3 RECTANGLES, do not train on these'}")
    print(f"  HSV decision image  : {args.hsv_source}")
    print("=" * 74)

    paths = resolve_paths(Path(args.manifest), Path(args.images_dir), args.n, args.seed)
    if not paths:
        print(f"[error] no images found under {args.images_dir}")
        return 1
    print(f"Sampling {len(paths)} image(s)\n")

    # The primary source is always first, because contact sheets and routing
    # records are written only for the first entry.
    configs = {args.hsv_source: FilterConfig(hsv_source=args.hsv_source)}
    if args.compare_hsv_source:
        other = "original" if args.hsv_source == "ssr" else "ssr"
        configs[other] = FilterConfig(hsv_source=other)

    selector = RegionalConfigurationSelector()
    stat_rows, routing_rows = [], []

    for path in paths:
        original = load_image(path)
        if original is None:
            print(f"[skip] unreadable: {path.name}")
            continue

        ssr_image = ssr_fn(original) if ssr_fn else reference_ssr(original)
        geometric = (
            seg_fn(original) if seg_fn else reference_region_masks(original.shape[:2])
        )

        for source_name, cfg in configs.items():
            patches = filter_all_regions(
                ssr_image=ssr_image,
                geometric_masks=geometric,
                config=cfg,
                reference_image=original,
            )
            stats = summarize(patches)

            for region, s in stats.items():
                stat_rows.append({
                    "filename": path.name,
                    "hsv_source": source_name,
                    "region": region,
                    **s,
                })

            # Only the primary source produces sheets and routing records,
            # so a comparison run does not double-write them.
            if source_name != list(configs)[0]:
                continue

            outputs = list(selector.route(patches, image_id=path.name))
            routing_rows.append(selector.routing_record(path.name, outputs))

            if not args.no_sheets:
                skin = {n: p.skin_mask for n, p in patches.items()}
                sheet = build_contact_sheet(original, ssr_image, geometric, skin, stats)
                cv2.imwrite(str(out_dir / f"{Path(path.name).stem}_qa.png"), sheet)

    if not stat_rows:
        print("[error] nothing processed")
        return 1

    stats_df = pd.DataFrame(stat_rows)
    routing_df = pd.DataFrame(routing_rows)

    results_dir = out_dir.parent
    stats_df.to_csv(results_dir / "phase7_coverage_stats.csv", index=False)
    routing_df.to_csv(results_dir / "phase7_routing_log.csv", index=False)

    # ---- report ---------------------------------------------------------
    print("RETENTION BY REGION (fraction of polygon surviving HSV filtering)")
    print("-" * 74)
    pivot = stats_df.pivot_table(
        index="region", columns="hsv_source", values="retention", aggfunc="mean"
    ).reindex(REGION_ORDER)
    print(pivot.round(3).to_string())

    print("\nFILTER TIER USAGE")
    print("-" * 74)
    tier = (
        stats_df.groupby(["hsv_source", "status"]).size().unstack(fill_value=0)
    )
    print(tier.to_string())

    print("\nROUTING (Phase 7.5)")
    print("-" * 74)
    print(f"  images routed    : {selector.stats['images_routed']}")
    print(f"  configs emitted  : {selector.stats['configs_emitted']}")
    print(f"  configs skipped  : {selector.stats['configs_skipped']}")
    if "full_face_regions_present" in routing_df.columns:
        complete = int((routing_df["full_face_regions_present"] == 5).sum())
        print(f"  full-face vectors with all 5 regions: {complete}/{len(routing_df)}")

    print(f"\nWrote {results_dir / 'phase7_coverage_stats.csv'}")
    print(f"Wrote {results_dir / 'phase7_routing_log.csv'}")
    if not args.no_sheets:
        print(f"Wrote contact sheets to {out_dir}/")

    print("\nWHAT TO LOOK FOR IN THE SHEETS")
    print("  - panel 3: do the polygons sit on the right anatomy? Check left/right")
    print("    cheeks are not mirrored — 'left' means the subject's left.")
    print("  - panel 4: eyebrows, lips and hair should be GONE, cheek and forehead")
    print("    skin should be largely intact. Heavy erosion of a cheek means the")
    print("    thresholds are wrong for your data, not that the face is unusual.")
    print("  - retention below ~0.4 on cheeks across most images is the signal to")
    print("    re-run with --compare-hsv-source before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
