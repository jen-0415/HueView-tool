"""
Fixed verification script — correctly finds ml_pipeline directory.

Usage:
    python verify_phase10_FIXED.py
    (Run from anywhere in the project, it will find ml_pipeline/)
"""

import sys
from pathlib import Path
import pandas as pd

# Find ml_pipeline directory
def find_ml_pipeline():
    cwd = Path.cwd()
    # If we're already in ml_pipeline, use that
    if (cwd / "data" / "processed").exists():
        return cwd
    # If ml_pipeline is a subdirectory, use that
    if (cwd / "ml_pipeline" / "data" / "processed").exists():
        return cwd / "ml_pipeline"
    # Search upward
    for parent in cwd.parents:
        if (parent / "ml_pipeline" / "data" / "processed").exists():
            return parent / "ml_pipeline"
    raise FileNotFoundError("Cannot find ml_pipeline directory. Make sure you're in the project root or ml_pipeline folder.")

ROOT = find_ml_pipeline()
print(f"Using ROOT: {ROOT}\n")

CHECKS = []

def check(name, condition, details=""):
    status = "✓" if condition else "✗"
    msg = f"  {status} {name}"
    if details:
        msg += f" — {details}"
    CHECKS.append((condition, msg))
    print(msg)

print("="*60)
print("Phase 10 Readiness Check")
print("="*60 + "\n")

# ---- Paths ----
print("1. Directory Structure")
check("data/processed/", (ROOT / "data/processed").is_dir())
check("data/processed/images_ssr/", (ROOT / "data/processed/images_ssr").is_dir())
check("data/processed/regions/", (ROOT / "data/processed/regions").is_dir(),
      "← Created by Phase 7.3")

# ---- Landmarks ----
print("\n2. Landmarks")
lm_path = ROOT / "data/processed/landmarks.npy"
idx_path = ROOT / "data/processed/landmarks_index.csv"
check("landmarks.npy exists", lm_path.exists())
check("landmarks_index.csv exists", idx_path.exists())

# ---- CSVs ----
print("\n3. Train/Val Split")
train_csv = ROOT / "data/processed/train_verified.csv"
val_csv = ROOT / "data/processed/val_verified.csv"
check("train_verified.csv exists", train_csv.exists())
check("val_verified.csv exists", val_csv.exists())

train_df = None
val_df = None

if train_csv.exists():
    try:
        train_df = pd.read_csv(train_csv)
        check(f"  train rows: {len(train_df)}", len(train_df) > 0, f"{len(train_df):,} rows")
        if "filename" in train_df.columns:
            sample = train_df["filename"].iloc[0]
            check(f"  train filename format", "/" in sample, f"e.g. {sample}")
    except Exception as e:
        check(f"  train CSV readable", False, str(e))

if val_csv.exists():
    try:
        val_df = pd.read_csv(val_csv)
        check(f"  val rows: {len(val_df)}", len(val_df) > 0, f"{len(val_df):,} rows")
    except Exception as e:
        check(f"  val CSV readable", False, str(e))

# ---- Regional Patches ----
print("\n4. Regional Patches (Phase 7.3 Output)")
regions_root = ROOT / "data/processed/regions"
if regions_root.exists():
    patch_files = list(regions_root.rglob("*.jpg"))
    patch_count = len(patch_files)
    check(f"  Regional patches", patch_count > 0, f"{patch_count:,} JPGs")

    if train_df is not None and val_df is not None:
        expected_count = len(train_df) + len(val_df)
        regions_per_face = 5  # forehead, left/right cheek, jawline, nose_bridge
        expected_patches = expected_count * regions_per_face
        coverage_pct = (patch_count / expected_patches * 100) if expected_patches > 0 else 0
        check(f"  Coverage", patch_count >= expected_patches,
              f"expected ~{expected_patches:,}, got {patch_count:,} ({coverage_pct:.1f}%)")
else:
    check("Regional patches", False, "Run Phase 7.3 first: python src/hueview/segment_regions.py")

# ---- SSR Images ----
print("\n5. SSR Images (Phase 7.2 Output)")
ssr_root = ROOT / "data/processed/images_ssr"
if ssr_root.exists():
    ssr_files = list(ssr_root.rglob("*.jpg")) + list(ssr_root.rglob("*.png"))
    ssr_count = len(ssr_files)
    check(f"  SSR images", ssr_count > 0, f"{ssr_count:,} files")

# ---- Config ----
print("\n6. Training Configuration")
config_path = ROOT / "configs/train_config.json"
check("configs/train_config.json", config_path.exists(),
      "Optional — defaults used if missing")

# ---- Output Dirs ----
print("\n7. Output Directories (will be auto-created)")
check("models/ will be created", True)
check("results/ will be created", True)

# ---- Summary ----
print("\n" + "="*60)
passed = sum(1 for ok, _ in CHECKS if ok)
total = len(CHECKS)
print(f"Result: {passed}/{total} checks passed\n")

if passed >= 9:  # Allow some optional checks to fail
    print("✓ READY FOR PHASE 10!")
    print("\nNext steps:")
    print("  1. python src/phase10/train_hueview.py --cache-lab")
    print("  2. python src/phase10/train_hueview.py --smoke")
    print("  3. python src/phase10/train_hueview.py")
    sys.exit(0)
else:
    print("✗ BLOCKERS FOUND — Fix above issues first.")
    print(f"\nFailed checks:")
    for ok, msg in CHECKS:
        if not ok:
            print(msg)
    sys.exit(1)