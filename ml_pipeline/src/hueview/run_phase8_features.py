"""
Phase 8 -- Full-Dataset Hybrid Feature Extraction (batch run, stage 1)
Manuscript: Chapter 3, Stage 4 (EfficientNet CNN Feature Extraction),
Stage 5 (RGB -> CIELAB), Fused Classification & Output; Data Generation
(Feature Selection, third step).

For every landmarked image in train/val/test, and for each of the five
regions (processed one region at a time, as the manuscript describes):

    8.1  CNN   : region's SSR patch (skin mask, zeros elsewhere, 224x224)
                 -> EfficientNetB0 (ImageNet, frozen)      -> 1280 values
    8.2  CIELAB: mean [L*, a*, b*] over the region's skin pixels -> 3 values

and saves them for 8.3 / Phase 10 training:

    ml_pipeline/data/processed/phase8_features/<split>_<region>_cnn.npy   (N, 1280) float32
    ml_pipeline/data/processed/phase8_features/<split>_<region>_lab.npy   (N, 3)    float32
    ml_pipeline/data/processed/phase8_features/<split>_<region>_index.csv
        row, filename, scc_label, scc_index, status, n_valid_px
    ml_pipeline/data/processed/phase8_features/skipped.csv
        filename, split, region, reason          (regions with no usable skin)

Row i of the three files for a (split, region) always describes the same
image, so fused input = concat(cnn[i], lab[i]) = 1283 values.

STAGE 1 = FROZEN BACKBONE. The manuscript's per-region EfficientNetB0s all
start from the same ImageNet weights; while frozen they are identical, so one
frozen instance gives exactly the vectors each region's own copy would. The
per-region copies only diverge in stage 2 (fine-tuning), which is training.

Augmentation (flip / rotation / colour jitter, training split only) is NOT
applied here -- it belongs to the training phase.

Inputs (from Phase 7):
    ml_pipeline/data/processed/{train,val,test}.csv      -- filename, SCC_label
    ml_pipeline/data/processed/landmarks_index.csv       -- landmarked images
    ml_pipeline/data/processed/label_maps/<filename>.png -- 7.4 masks
    ml_pipeline/data/processed/images_ssr/<filename>     -- 7.2 corrected SSR

Run from the HueView-tool repo root:
    python -m ml_pipeline.src.hueview.run_phase8_features            # full run
    python -m ml_pipeline.src.hueview.run_phase8_features --limit 50 # quick test
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------ project root --


def find_project_root(start: Path, marker: str = "ml_pipeline") -> Path:
    """Walk upward from `start` until a directory containing `marker` is found."""
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
        if candidate.name == marker:
            return candidate.parent
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.src.hueview.regions import REGION_ORDER, decode_label_map  # noqa: E402
from ml_pipeline.src.hueview.region_selector import RegionalConfigurationSelector  # noqa: E402
from ml_pipeline.src.hueview.cnn_features import (  # noqa: E402
    CNN_DIM, LAB_DIM, build_backbone, extract_features, prepare_patch)
from ml_pipeline.src.hueview.cielab_features import build_cielab_vector  # noqa: E402

# ---------------------------------------------------------------- config --

DATA_PROCESSED = PROJECT_ROOT / "ml_pipeline" / "data" / "processed"
SPLITS = ("train", "val", "test")
INDEX_PATH = DATA_PROCESSED / "landmarks_index.csv"
LABEL_MAPS_ROOT = DATA_PROCESSED / "label_maps"
SSR_ROOT = DATA_PROCESSED / "images_ssr"
OUT_DIR = DATA_PROCESSED / "phase8_features"

BATCH_SIZE = 64                      # images per EfficientNetB0 call
SCC_CLASSES = tuple(f"SCC-{i}" for i in range(1, 7))

# ----------------------------------------------------------------- helpers --


def load_split(split: str, landmarked: set) -> pd.DataFrame:
    """filename + SCC label for one split, restricted to landmarked images."""
    path = DATA_PROCESSED / f"{split}.csv"
    if not path.is_file():
        raise SystemExit(f"ERROR: {path} not found.")
    df = pd.read_csv(path)
    label_col = next((c for c in df.columns if c.lower() in ("scc_label", "scc")), None)
    if label_col is None:
        raise SystemExit(f"ERROR: no SCC label column in {path}. Columns: {list(df.columns)}")
    df = df[["filename", label_col]].rename(columns={label_col: "scc_label"})
    df["scc_label"] = df["scc_label"].astype(str)
    bad = sorted(set(df["scc_label"]) - set(SCC_CLASSES))
    if bad:
        raise SystemExit(f"ERROR: unexpected SCC labels in {path}: {bad[:5]}")
    df["scc_index"] = df["scc_label"].map({c: i for i, c in enumerate(SCC_CLASSES)})
    before = len(df)
    df = df[df["filename"].isin(landmarked)].reset_index(drop=True)
    print(f"  {split:5s}: {len(df)} landmarked images (of {before} in {path.name})")
    return df


def load_regions(filename: str):
    """Decode one image's Phase 7.4 label map on its SSR image -> {region: config}."""
    label = cv2.imread(str(LABEL_MAPS_ROOT / Path(filename).with_suffix(".png")),
                       cv2.IMREAD_UNCHANGED)
    ssr_bgr = cv2.imread(str(SSR_ROOT / filename))
    if label is None or ssr_bgr is None:
        return None
    ssr_rgb = cv2.cvtColor(ssr_bgr, cv2.COLOR_BGR2RGB)
    patches = decode_label_map(label, ssr_rgb)
    selector = RegionalConfigurationSelector(configurations=REGION_ORDER,
                                             skip_unusable=False)
    return selector.route_all(patches, image_id=filename)


# -------------------------------------------------------------------- main --


def run(limit: int | None, weights: str | None):
    print(f"Project root: {PROJECT_ROOT}")
    for p in (INDEX_PATH, LABEL_MAPS_ROOT, SSR_ROOT):
        if not p.exists():
            raise SystemExit(f"ERROR: {p} not found -- run Phase 7 first.")

    landmarked = set(pd.read_csv(INDEX_PATH)["filename"].astype(str))
    print(f"{INDEX_PATH.name}: {len(landmarked)} landmarked images")
    splits = {s: load_split(s, landmarked) for s in SPLITS}
    if limit:
        splits = {s: d.head(limit) for s, d in splits.items()}
        print(f"LIMIT set -- {limit} images per split")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading EfficientNetB0 (weights={weights}, frozen) ...")
    backbone = build_backbone(weights=weights)

    skipped = []
    t0 = time.time()
    for split, df in splits.items():
        # per-region accumulators
        cnn = {r: [] for r in REGION_ORDER}
        lab = {r: [] for r in REGION_ORDER}
        idx = {r: [] for r in REGION_ORDER}
        pending = {r: [] for r in REGION_ORDER}   # (row_meta, patch, lab_vec)

        def flush(region):
            if not pending[region]:
                return
            patches = np.stack([p for _, p, _ in pending[region]])
            vecs = extract_features(backbone, patches, batch_size=BATCH_SIZE)
            for (meta, _, lab_vec), vec in zip(pending[region], vecs):
                cnn[region].append(vec)
                lab[region].append(lab_vec)
                idx[region].append(meta)
            pending[region].clear()

        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Phase 8 [{split}]"):
            configs = load_regions(row.filename)
            if configs is None:
                for r in REGION_ORDER:
                    skipped.append((row.filename, split, r, "label_map_or_ssr_missing"))
                continue
            for r in REGION_ORDER:
                cfg = configs.get(r)
                patch = prepare_patch(cfg.image, cfg.mask) if cfg is not None else None
                lab_vec = build_cielab_vector(cfg) if cfg is not None else None
                if patch is None or lab_vec is None:
                    skipped.append((row.filename, split, r, "no_usable_skin"))
                    continue
                meta = {"filename": row.filename, "scc_label": row.scc_label,
                        "scc_index": int(row.scc_index), "status": cfg.status,
                        "n_valid_px": int(cfg.n_valid_px)}
                pending[r].append((meta, patch, lab_vec.astype(np.float32)))
                if len(pending[r]) >= BATCH_SIZE:
                    flush(r)

        for r in REGION_ORDER:
            flush(r)
            c = np.asarray(cnn[r], np.float32).reshape(-1, CNN_DIM)
            l = np.asarray(lab[r], np.float32).reshape(-1, LAB_DIM)
            i = pd.DataFrame(idx[r])
            assert len(c) == len(l) == len(i), f"row mismatch for {split}/{r}"
            np.save(OUT_DIR / f"{split}_{r}_cnn.npy", c)
            np.save(OUT_DIR / f"{split}_{r}_lab.npy", l)
            i.insert(0, "row", range(len(i)))
            i.to_csv(OUT_DIR / f"{split}_{r}_index.csv", index=False)
        print(f"  saved {split}: " + ", ".join(f"{r}={len(cnn[r])}" for r in REGION_ORDER))

    pd.DataFrame(skipped, columns=["filename", "split", "region", "reason"]).to_csv(
        OUT_DIR / "skipped.csv", index=False)

    # ------------------------------------------------------------ checkpoint --
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Saved to {OUT_DIR}")
    print("\nPHASE 8 CHECKPOINT (fused width per region = CNN + CIELAB)")
    print("-" * 60)
    ok = True
    for split in splits:
        for r in REGION_ORDER:
            c = np.load(OUT_DIR / f"{split}_{r}_cnn.npy", mmap_mode="r")
            l = np.load(OUT_DIR / f"{split}_{r}_lab.npy", mmap_mode="r")
            fused = c.shape[1] + l.shape[1]
            good = c.shape[1] == CNN_DIM and l.shape[1] == LAB_DIM and len(c) == len(l)
            ok &= good and bool(np.isfinite(c).all()) and bool(np.isfinite(l).all())
            print(f"  {split:5s} {r:12s} n={len(c):6d}  {c.shape[1]} + {l.shape[1]} = {fused}"
                  f"  {'ok' if good else 'MISMATCH'}")
    print(f"\nSkipped regions (no usable skin / missing): {len(skipped)}"
          f"  (logged to {OUT_DIR / 'skipped.csv'})")
    print("Checkpoint:", "PASSED -- every region fuses to 1283" if ok else "FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 8 feature extraction (stage 1)")
    ap.add_argument("--limit", type=int, default=None, help="images per split (quick test)")
    ap.add_argument("--no-imagenet", action="store_true",
                    help="random weights (testing only -- never for real features)")
    a = ap.parse_args()
    run(a.limit, None if a.no_imagenet else "imagenet")
