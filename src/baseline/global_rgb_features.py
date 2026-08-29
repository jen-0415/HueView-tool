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
    from path_resolver import resolve_image_path
except ImportError:
    from src.baseline.path_resolver import resolve_image_path


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
            img_path, rule = Path(lookup[fname]), "table"
        else:
            img_path, rule, _root = resolve_image_path(fname)
        rule_counts[rule] = rule_counts.get(rule, 0) + 1

        if img_path is None:
            failed.append((fname, "no file on disk matches this entry"))
            continue

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
    # Image roots are defined in path_resolver.DEFAULT_ROOTS / V5_ROOT.
    MANIFEST = "data/processed/train.csv"  # splits are already frozen

    manifest = pd.read_csv(MANIFEST)
    sample = manifest.head(5)

    print("Running 6.1 on a 5-image sample first.")
    print("Sanity check: r/g/b means should land roughly 0.3-0.8 for normally")
    print("exposed skin — values pinned at 0.0 or 1.0 usually mean a bad crop")
    print("or a path pointing at the wrong file.\n")

    sample_result = batch_extract(sample)
    print(sample_result)

    # Once the sample looks right, run it for real on each frozen split and
    # save the output — Phase 6.2 will load these CSVs to concatenate with
    # the CNN feature vector:
    #
    # for split in ["train", "val", "test"]:
    #     feats = batch_extract(f"data/processed/{split}.csv")
    #     feats.to_csv(f"data/processed/baseline_rgb_{split}.csv", index=False)
