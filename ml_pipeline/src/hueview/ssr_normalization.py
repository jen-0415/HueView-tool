"""
SSR (Single-Scale Retinex) Illumination Normalization

Phase 7, Stream B (parallel to 7.1) -- HueView

SSR is used to remove uneven illumination across the face. The SSR
reflectance is computed on the luminance channel using a Gaussian
illumination estimate with sigma=50, and the image is re-lit under a
uniform illumination equal to its mean illumination.

The resulting correction is applied equally to the R, G and B channels
(so skin hue/chroma is kept), and the WHOLE image's mean intensity is
preserved. The face's own brightness is not: a face brighter than its
surroundings (e.g. framed by dark hair) is darkened, a face darker than
its surroundings is brightened.

Job list comes from landmarks_index.csv (Phase 7.1's output) -- only
images that successfully got landmarks are processed.

Images are looked up directly under data/processed/images/MST-N/.
Output mirrors that MST-N/ structure under data/processed/images_ssr/.

Paths are anchored to <project_root> (the folder containing "data/"), found
by walking up from this file's own location.    
"""

import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------------------------------------------------------ project root --

def find_project_root(start: Path, marker: str = "data") -> Path:
    """Walk upward until a directory containing `marker` is found."""
    for candidate in [start] + list(start.parents):
        if (candidate / marker).is_dir():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


# ---------------------------------------------------------------- config --

IMAGES_ROOT = PROJECT_ROOT / "data" / "processed" / "images"
LANDMARKS_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "landmarks_index.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "images_ssr"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 50 (was 30): at 30 the blur is small enough to treat "face brighter than its
# hair/background" as uneven lighting and darken the face (-12..-21% on the
# ssr_test photos); 50 roughly halves that. data/processed/images_ssr/ and the
# trained HueView models were made at an earlier setting -- re-run this batch
# and retrain for them to match the web app.
SIGMA = 50.0
EPSILON = 1.0
STRENGTH = 1.0            # 0 = no effect, 1 = full flattening
GAIN_LIMITS = (0.5, 2.0)  # guards against extreme corrections
SAMPLE_SIZE = 0


# ------------------------------------------------------------------- SSR --

def apply_ssr(
    image: np.ndarray,
    sigma: float = SIGMA,
    epsilon: float = EPSILON,
    strength: float = STRENGTH,
    gain_limits: tuple = GAIN_LIMITS,
) -> np.ndarray:
    """
    SSR-based illumination normalization.

        Y(x,y)  = 0.299 R + 0.587 G + 0.114 B          (luminance)
        L(x,y)  = G_sigma * Y                          (illumination)
        r(x,y)  = log Y(x,y) - log L(x,y)              (SSR reflectance)
        Y'(x,y) = exp(r(x,y)) * mean(L)                (uniform re-lighting)
        g(x,y)  = clip(Y'/Y, 0.5, 2.0), rescaled so mean(g*Y) = mean(Y)
        I'_c    = g * I_c   for c in {R, G, B}
    """

    img = image.astype(np.float64) + epsilon

    # Luminance (OpenCV loads images as BGR)
    lum = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]

    # Estimate illumination
    illumination = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # SSR reflectance
    retinex = np.log(lum) - np.log(illumination)

    # Re-light under a uniform illumination equal to the mean illumination
    target = illumination.mean()
    gain = (np.exp(retinex) * target) / lum

    # Prevent extreme corrections
    gain = np.clip(gain ** strength, *gain_limits)

    # Preserve the original overall brightness
    gain *= lum.mean() / (lum * gain).mean()

    # Apply the same gain to all three channels
    corrected = img * gain[:, :, None]

    return np.clip(corrected - epsilon, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- worker --

def process_one(paths):
    src_path, dst_path = paths

    try:
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)

        if img is None:
            return (str(src_path), "unreadable")

        result = apply_ssr(img)

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(str(dst_path), result)

        if not ok:
            return (str(src_path), "write failed")

        return None

    except Exception as e:
        return (str(src_path), repr(e))


# ------------------------------------------------------------- job list --

def collect_jobs():
    """
    Job list = every image in landmarks_index.csv, found under
    data/processed/images/MST-N/.

    Matching order:
      1. path   -- exact relative path (e.g. "MST-3/x.jpg")
      2. copy   -- "x (2).jpg" in the CSV renamed to "x_1.jpg" on disk,
                   same MST folder
      3. name   -- exact filename, any folder
      4. lower  -- filename, ignoring case
      5. stem   -- filename without extension, ignoring case (.png vs .jpg)
    Rules 3-5 are used only if they point to exactly one file.
    """

    if not LANDMARKS_INDEX_PATH.is_file():
        raise SystemExit(
            f"ERROR: {LANDMARKS_INDEX_PATH} not found -- "
            f"run landmark_extraction.py (7.1) first."
        )

    if not IMAGES_ROOT.is_dir():
        raise SystemExit(f"ERROR: {IMAGES_ROOT} not found.")

    by_name, by_lower, by_stem = {}, {}, {}

    for p in IMAGES_ROOT.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            by_name.setdefault(p.name, []).append(p)
            by_lower.setdefault(p.name.lower(), []).append(p)
            by_stem.setdefault(p.stem.lower(), []).append(p)

    print(f"Found {sum(len(v) for v in by_name.values())} images under {IMAGES_ROOT}")

    def unique(table, key):
        hits = table.get(key, [])
        return hits[0] if len(hits) == 1 else None

    index_df = pd.read_csv(LANDMARKS_INDEX_PATH)

    jobs, unresolved = [], []
    counts = {"path": 0, "copy": 0, "name": 0, "lower": 0, "stem": 0}

    for filename in index_df["filename"].astype(str):
        name = Path(filename).name
        direct = IMAGES_ROOT / filename

        # "x (2).jpg" in CSV was renamed to "x_1.jpg" on disk
        renamed = re.sub(
            r" \((\d+)\)(?=\.\w+$)",
            lambda m: f"_{int(m.group(1)) - 1}",
            filename,
        )

        if direct.is_file():
            src_path, how = direct, "path"
        elif renamed != filename and (IMAGES_ROOT / renamed).is_file():
            src_path, how = IMAGES_ROOT / renamed, "copy"
        elif (p := unique(by_name, name)):
            src_path, how = p, "name"
        elif (p := unique(by_lower, name.lower())):
            src_path, how = p, "lower"
        elif (p := unique(by_stem, Path(name).stem.lower())):
            src_path, how = p, "stem"
        else:
            unresolved.append(filename)
            continue

        counts[how] += 1
        # Save under the landmarks_index name so downstream phases find it
        dst_path = OUTPUT_ROOT / filename
        jobs.append((src_path, dst_path))

    print(f"Matched by: {counts}")

    return jobs, unresolved


# ------------------------------------------------------------------- main --

def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"SSR sigma: {SIGMA}")

    all_jobs, unresolved = collect_jobs()

    print(
        f"{LANDMARKS_INDEX_PATH.name}: "
        f"{len(all_jobs)} resolved, "
        f"{len(unresolved)} unresolved"
    )

    if unresolved:
        log_path = PROJECT_ROOT / "ssr_unresolved.log"
        log_path.write_text("\n".join(unresolved))
        print(f"Unresolved filenames logged to {log_path}")

    if SAMPLE_SIZE:
        all_jobs = all_jobs[:SAMPLE_SIZE]
        print(
            f"SAMPLE_SIZE set -- "
            f"only processing {len(all_jobs)} images"
        )

    print(
        f"Total: {len(all_jobs)} images, "
        f"{cpu_count()} workers"
    )

    failures = []

    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(
            pool.imap_unordered(process_one, all_jobs),
            total=len(all_jobs)
        ):
            if result is not None:
                failures.append(result)

    print(
        f"Finished. {len(failures)} failed "
        f"out of {len(all_jobs)}."
    )

    if failures:
        log_path = PROJECT_ROOT / "ssr_failures.log"

        with open(log_path, "w") as f:
            for path, err in failures:
                f.write(f"{path}\t{err}\n")

        print(f"Failures logged to {log_path}")


if __name__ == "__main__":
    main()