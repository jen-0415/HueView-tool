"""
Phase 7.4 -- Full-Dataset HSV Skin Filtering (batch run)

Runs the frozen Phase 7.4 methodology (configs/hsv_skin_thresholds.json,
hsv_source="original") across the full landmarked dataset.

Packs geometric & skin masks via encode_label_map() into ONE PNG per image.
"""

import os
import re
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

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

from ml_pipeline.src.baseline.path_resolver import resolve_image_path  # noqa: E402
from ml_pipeline.src.hueview.segment_regions import REGIONS, build_region_mask  # noqa: E402
from ml_pipeline.src.hueview.hsv_skin_filter import FilterConfig, filter_all_regions, summarize  # noqa: E402
from ml_pipeline.src.hueview.regions import encode_label_map  # noqa: E402

# ---------------------------------------------------------------- config --

DATA_PROCESSED = PROJECT_ROOT / "ml_pipeline" / "data" / "processed"
IMAGES_ROOT = DATA_PROCESSED / "images"
SSR_ROOT = DATA_PROCESSED / "images_ssr"

LANDMARKS_PATH = DATA_PROCESSED / "landmarks.npy"
INDEX_PATH = DATA_PROCESSED / "landmarks_index.csv"
LABEL_MAPS_ROOT = DATA_PROCESSED / "label_maps"
CONFIG_PATH = PROJECT_ROOT / "ml_pipeline" / "configs" / "hsv_skin_thresholds.json"
STATS_PATH = DATA_PROCESSED / "phase7_4_full_coverage_stats.csv"
FAILURES_PATH = DATA_PROCESSED / "phase7_4_failures.csv"

LIMIT = None  # None ensures all ~43k images are processed

# Global lookup tables shared across worker tasks
LOOKUP_TABLES = (None, None, None)
CONFIG_OBJ = None

# ----------------------------------------------------------- lookup maps --


def build_image_lookup_table() -> tuple[dict[str, Path], dict[str, list[Path]], dict[str, Path]]:
    """Indexes IMAGES_ROOT with multi-level fallbacks."""
    print(f"Indexing pre-SSR images under {IMAGES_ROOT}...")
    exact_map = {}
    name_to_paths = {}
    stem_map = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if IMAGES_ROOT.is_dir():
        for root, _, files in os.walk(IMAGES_ROOT):
            root_path = Path(root)
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in valid_exts:
                    full_path = root_path / file

                    try:
                        rel_mst = full_path.relative_to(IMAGES_ROOT)
                        exact_map[str(rel_mst).replace("\\", "/")] = full_path
                        exact_map[str(rel_mst)] = full_path
                    except ValueError:
                        pass

                    file_lower = file.lower()
                    if file_lower not in name_to_paths:
                        name_to_paths[file_lower] = []
                    name_to_paths[file_lower].append(full_path)

                    stem_map[Path(file).stem.lower()] = full_path

    print(f"Indexed {len(exact_map)} relative paths and {len(name_to_paths)} unique filenames.")
    return exact_map, name_to_paths, stem_map


def get_path_variants(file_str: str) -> list[str]:
    """Generates clean filename variants and swaps extensions (.png <-> .bmp <-> .jpg)."""
    clean_str = re.sub(r"\s*\(\d+\)", "", file_str)
    variants = [file_str, clean_str]
    
    extended_variants = []
    for var in variants:
        p = Path(var)
        extended_variants.append(var)
        for ext in [".bmp", ".jpg", ".png", ".jpeg"]:
            if p.suffix.lower() != ext:
                extended_variants.append(str(p.with_suffix(ext)))

    # Preserve order while removing duplicates
    return list(dict.fromkeys(extended_variants))


def resolve_local_path(filename: str, exact_map: dict, name_to_paths: dict, stem_map: dict):
    """Resolves local paths across MST subfolders, stripping copy suffixes and testing extensions."""
    str_file = str(filename).replace("\\", "/")
    if "images/" in str_file:
        str_file = str_file.split("images/")[-1]

    variants = get_path_variants(str_file)

    for var in variants:
        if var in exact_map:
            return exact_map[var]

        bare_name = Path(var).name.lower()
        if bare_name in name_to_paths:
            return name_to_paths[bare_name][0]

        stem_name = Path(var).stem.lower()
        if stem_name in stem_map:
            return stem_map[stem_name]

        if f"{stem_name}_1" in stem_map:
            return stem_map[f"{stem_name}_1"]

    resolved, _, _ = resolve_image_path(filename)
    return resolved


def resolve_ssr_path(filename: str) -> Path | None:
    """Locates the SSR-normalized image across cohort directories and extension variants."""
    str_file = str(filename).replace("\\", "/")
    if "images/" in str_file:
        str_file = str_file.split("images/")[-1]

    variants = get_path_variants(str_file)

    # 1. Direct path check under SSR_ROOT
    for var in variants:
        cand = SSR_ROOT / var
        if cand.is_file():
            return cand

    # 2. Priority cohort scan
    cohorts = ["processed", "v5_processed", "c1_processed", "c2_processed"]
    for cohort in cohorts:
        for var in variants:
            cand = SSR_ROOT / cohort / var
            if cand.is_file():
                return cand
            
            # Subfolder duplicate prefix check
            v_path = Path(var)
            cand_sub = SSR_ROOT / cohort / v_path
            if cand_sub.is_file():
                return cand_sub

    # 3. Fallback scan by bare filename
    bare_names = list(set(Path(v).name.lower() for v in variants))
    for bare_name in bare_names:
        matched = list(SSR_ROOT.rglob(bare_name))
        if matched and matched[0].is_file():
            return matched[0]

    return None


# --------------------------------------------------------------- worker --


def init_worker(config_data, exact_map, name_to_paths, stem_map):
    """Worker process initializer to store lookup objects globally per process."""
    global CONFIG_OBJ, LOOKUP_TABLES
    CONFIG_OBJ = config_data
    LOOKUP_TABLES = (exact_map, name_to_paths, stem_map)


def process_one(args):
    filename, row_index, landmarks_px = args
    exact_map, name_to_paths, stem_map = LOOKUP_TABLES

    # 1. Resolve SSR image path
    ssr_path = resolve_ssr_path(filename)
    if ssr_path is None or not ssr_path.is_file():
        return (filename, None, "ssr_image_missing_or_unreadable")

    ssr_bgr = cv2.imread(str(ssr_path))
    if ssr_bgr is None:
        return (filename, None, "ssr_image_missing_or_unreadable")

    ssr_rgb = cv2.cvtColor(ssr_bgr, cv2.COLOR_BGR2RGB)

    geometric_masks = {
        name: build_region_mask(landmarks_px, idx, ssr_rgb.shape).astype(bool)
        for name, idx in REGIONS.items()
    }

    reference_rgb = None
    if CONFIG_OBJ.hsv_source == "original":
        orig_path = resolve_local_path(filename, exact_map, name_to_paths, stem_map)
        if orig_path is None or not Path(orig_path).is_file():
            return (filename, None, "original_not_found")

        orig_bgr = cv2.imread(str(orig_path))
        if orig_bgr is None:
            return (filename, None, "original_unreadable")
        reference_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    patches = filter_all_regions(
        ssr_image=ssr_rgb,
        geometric_masks=geometric_masks,
        config=CONFIG_OBJ,
        reference_image=reference_rgb,
    )

    label_map = encode_label_map(patches, ssr_rgb.shape[:2])
    
    # Save encoded label map using clean path
    str_file_clean = re.sub(r"\s*\(\d+\)", "", str(filename).replace("\\", "/"))
    if "images/" in str_file_clean:
        str_file_clean = str_file_clean.split("images/")[-1]
        
    out_path = LABEL_MAPS_ROOT / Path(str_file_clean).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(out_path), label_map)
    if not ok:
        return (filename, None, "label_map_write_failed")

    stats = summarize(patches)
    rows = [{"filename": filename, "region": region, **s} for region, s in stats.items()]
    return (filename, rows, None)


# --------------------------------------------------------------- main --


def run():
    print(f"Project root: {PROJECT_ROOT}")

    if not LANDMARKS_PATH.is_file() or not INDEX_PATH.is_file():
        raise SystemExit(
            "ERROR: landmarks.npy / landmarks_index.csv not found -- run landmark_extraction.py (7.1) first."
        )
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"ERROR: {CONFIG_PATH} not found -- freeze the Phase 7.4 config first.")

    config = FilterConfig.from_json(CONFIG_PATH)
    print(f"Using frozen config: {CONFIG_PATH} (hsv_source={config.hsv_source})")

    exact_map, name_to_paths, stem_map = build_image_lookup_table()

    landmarks_array = np.load(LANDMARKS_PATH)
    index_df = pd.read_csv(INDEX_PATH)
    print(f"Loaded {len(index_df)} landmark rows, landmarks array shape {landmarks_array.shape}")

    jobs = [
        (row["filename"], row["row_index"], landmarks_array[row["row_index"]])
        for _, row in index_df.iterrows()
    ]

    if LIMIT:
        jobs = jobs[:LIMIT]
        print(f"LIMIT set -- only processing {len(jobs)} images")

    print(f"Total: {len(jobs)} images, {cpu_count()} workers")

    all_rows = []
    failures = []

    with Pool(
        processes=cpu_count(),
        initializer=init_worker,
        initargs=(config, exact_map, name_to_paths, stem_map),
    ) as pool:
        for filename, rows, error in tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs), desc="Phase 7.4"):
            if error is not None:
                failures.append({"filename": filename, "reason": error})
            else:
                all_rows.extend(rows)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(STATS_PATH, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)

    n_success = len(jobs) - len(failures)
    print(f"\nProcessed {n_success} / {len(jobs)} images successfully")
    print(f"Failed: {len(failures)} (logged to {FAILURES_PATH.name})" if failures else "Failed: 0")
    print(f"Saved label maps to {LABEL_MAPS_ROOT}")
    if all_rows:
        print(f"Saved coverage stats to {STATS_PATH}")
        stats_df = pd.DataFrame(all_rows)
        print("\nRETENTION BY REGION (this run)")
        print(
            stats_df.pivot_table(index="region", values="retention", aggfunc="mean")
            .round(3)
            .to_string()
        )
        print("\nFILTER TIER USAGE (this run)")
        print(stats_df.groupby(["region", "status"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    run()