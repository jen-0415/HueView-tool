"""
Phase 6.1 — Global RGB Feature Extraction (Baseline Pipeline)
================================================================

For each preprocessed 224x224 face image, compute a 3-value global RGB
feature vector: the per-channel mean, normalized to [0, 1].

This is deliberately the "dumb" feature: no skin masking, no regional
split, no illumination correction. That's the whole point of Baseline —
it's the control condition Phase 6.2's plain EfficientNetB0 has to be
compared against, and HueView (Phases 7-9) is the version that adds
SSR + regional segmentation + CIELAB on top of this. If Baseline
already did any of that, the comparison in Phase 12 would be measuring
the wrong thing.

Feeds into: Phase 6.2 (concatenated with the CNN feature vector) and
Phase 6.3 (the rule-based undertone branch reuses this same vector).
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Manifest filenames resolve across four roots under three naming conventions.
# path_resolver is the shared logic — import it rather than reimplementing, so
# every phase reads the same bytes for the same manifest row.
try:
    from ml_pipeline.src.baseline.path_resolver import resolve_image_path
except ImportError:
    from ml_pipeline.src.baseline.path_resolver import resolve_image_path


def extract_global_rgb_features(image: np.ndarray) -> np.ndarray:
    """
    Compute the global RGB feature vector for one face image.

    Args:
        image: RGB image array, shape (H, W, 3), dtype uint8, values 0-255.
               Must already be in RGB order, not BGR — if you loaded it
               with cv2.imread, convert first (see extract_features_from_path).

    Returns:
        np.ndarray of shape (3,), dtype float32: [r_mean, g_mean, b_mean],
        each in [0, 1].
    """
    if image is None:
        raise ValueError("extract_global_rgb_features received a None image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {getattr(image, 'shape', None)}")

    img = image.astype(np.float32)

    r_mean = img[:, :, 0].mean() / 255.0
    g_mean = img[:, :, 1].mean() / 255.0
    b_mean = img[:, :, 2].mean() / 255.0

    return np.array([r_mean, g_mean, b_mean], dtype=np.float32)


def extract_features_from_path(image_path) -> np.ndarray:
    """
    Load a face image from disk and extract its global RGB feature vector.
    Handles the BGR->RGB conversion that cv2.imread requires.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return extract_global_rgb_features(rgb)


def batch_extract(manifest_csv, filename_col: str = "filename") -> pd.DataFrame:
    """
    Run 6.1 over every image listed in a manifest (e.g. train.csv, val.csv,
    or test.csv from Phase 5's frozen split).

    Args:
        manifest_csv: path to a manifest CSV, or an already-loaded DataFrame.
        filename_col: name of the column holding image filenames.

    Images that fail to load are logged and skipped rather than crashing
    the whole batch or being silently dropped — check the printed warnings
    before trusting the output, since a large failure count usually means
    a path or filename-column mismatch rather than genuinely bad images
    (those should already have been filtered out in Phase 2).

    Returns a DataFrame: filename, r_mean, g_mean, b_mean.
    """
    # Accept either a path to a CSV or an already-loaded DataFrame, so callers
    # can pass a slice (e.g. manifest.head(5)) without writing a temp file.
    if isinstance(manifest_csv, pd.DataFrame):
        manifest = manifest_csv
    else:
        manifest = pd.read_csv(manifest_csv)

    rows = []
    failed = []
    rule_counts = {}

    # Prefer the frozen resolution table if it exists. It was verified per-row
    # against each image's stored mean_Y, so it's more reliable than re-running
    # the routing rules — and it guarantees every phase reads identical bytes.
    lookup = {}
    table = Path("data/processed/resolved_manifest.csv")
    if table.exists():
        t = pd.read_csv(table)
        lookup = dict(zip(t["filename"], t["resolved_path"]))
        print(f"[6.1] Using frozen resolution table ({len(lookup)} entries).")

    for _, row in manifest.iterrows():
        fname = row[filename_col]

        if fname in lookup and isinstance(lookup[fname], str):
            img_path = Path(lookup[fname])
            rule = "table"
            # The table was built on a different machine or folder layout.
            # If the path doesn't exist on disk, fall back to the live resolver.
            if not img_path.is_file():
                img_path, rule, _root = resolve_image_path(fname)
                if img_path is None:
                    failed.append((fname, "not in table and not found by resolver"))
                    rule_counts["not_found"] = rule_counts.get("not_found", 0) + 1
                    continue
        else:
            img_path, rule, _root = resolve_image_path(fname)
            if img_path is None:
                failed.append((fname, "no file on disk matches this entry"))
                rule_counts["not_found"] = rule_counts.get("not_found", 0) + 1
                continue

        rule_counts[rule] = rule_counts.get(rule, 0) + 1

        try:
            r, g, b = extract_features_from_path(img_path)
            rows.append({"filename": fname, "r_mean": r, "g_mean": g, "b_mean": b})
        except (FileNotFoundError, ValueError) as e:
            failed.append((fname, str(e)))

    # Any row NOT coming from the table fell back to rule-based routing, which
    # is less reliable — worth knowing about rather than having it pass silently.
    non_table = {k: v for k, v in rule_counts.items() if k not in ("table", "not_found")}
    if non_table and lookup:
        summary = ", ".join(f"{v} {k}" for k, v in sorted(non_table.items()))
        print(f"[6.1] {sum(non_table.values())} row(s) not in the table, resolved by rule: {summary}")
    elif non_table:
        summary = ", ".join(f"{v} {k}" for k, v in sorted(non_table.items()))
        print(f"[6.1] Path rules used: {summary}")
        print("[6.1] Tip: run resolve_manifest.py to build the verified table first.")

    if failed:
        print(f"[6.1] {len(failed)} image(s) failed feature extraction:")
        for fname, err in failed[:10]:
            print(f"    {fname}: {err}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    result = pd.DataFrame(rows, columns=["filename", "r_mean", "g_mean", "b_mean"])
    print(f"[6.1] Extracted features for {len(result)}/{len(manifest)} images.")
    return result


if __name__ == "__main__":
    import sys
    import time

    # Image roots come from path_resolver; paths resolve via the frozen table.
    PROC = Path("data/processed")
    SPLITS = ["train", "val", "test"]

    print("=" * 66)
    print("PHASE 6.1 — GLOBAL RGB FEATURE EXTRACTION")
    print("=" * 66)

    # Sanity-check a handful before committing to ~43k images.
    first = pd.read_csv(PROC / "train.csv")
    print("\nSample of 5 from train.csv first.")
    print("Expect r/g/b roughly 0.2-0.8 for normally exposed skin. Values")
    print("pinned at 0.0 or 1.0 mean a bad crop or a wrong file.\n")
    print(batch_extract(first.head(5)).to_string(index=False))

    reply = input("\nLook right? Extract all three splits? [y/N] ").strip().lower()
    if reply != "y":
        print("Stopped. Nothing written.")
        sys.exit(0)

    total_start = time.time()
    for split in SPLITS:
        path = PROC / f"{split}.csv"
        if not path.exists():
            print(f"\n!! {path} not found, skipping.")
            continue

        df = pd.read_csv(path)
        print(f"\n{'-' * 66}")
        print(f"{split}: {len(df)} rows")
        print("-" * 66)

        t0 = time.time()
        feats = batch_extract(df)
        out = PROC / f"baseline_rgb_{split}.csv"
        feats.to_csv(out, index=False)

        print(f"[6.1] Wrote {out}  ({len(feats)} rows, {time.time() - t0:.0f}s)")
        if len(feats):
            desc = feats[["r_mean", "g_mean", "b_mean"]].describe().loc[
                ["min", "mean", "max"]]
            print(desc.to_string())

            # A channel mean at a hard 0 or 1 means the image is fully black or
            # fully saturated — worth catching now rather than after training.
            extreme = feats[
                (feats[["r_mean", "g_mean", "b_mean"]] <= 0.001).any(axis=1)
                | (feats[["r_mean", "g_mean", "b_mean"]] >= 0.999).any(axis=1)
            ]
            if len(extreme):
                print(f"\n  !! {len(extreme)} image(s) have a channel pinned at 0 or 1:")
                for fn in extreme["filename"].head(5):
                    print(f"       {fn}")

    print(f"\n{'=' * 66}")
    print(f"Done in {time.time() - total_start:.0f}s.")
    print("Next: python src/baseline/undertone.py  (reads these CSVs)")
