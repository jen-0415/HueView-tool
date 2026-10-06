"""
SSR Comparison -- HueView, Phase 7 Stream B check

Side-by-side ng:
    [ Original | Old SSR (I * I/L) | New SSR (I * target/L) | Gain map ]

Para makita kung talagang napapantay ng bagong version ang lighting
sa loob ng mukha, at kung hindi nagagalaw ang kulay ng balat.

Usage:
    python ssr_compare.py                     # random samples per MST folder
    python ssr_compare.py path/a.jpg b.jpg    # specific images

Output (under <project_root>/data/processed/ssr_comparison/):
    <MST-N>__<name>.png   -- 4-panel comparison per image
    metrics.csv           -- numbers per image (see METRICS below)

METRICS (computed on luminance Y, per version):
    mean_Y       -- overall brightness. Dapat halos pareho sa original
                    (hindi dapat binabago ang skin-tone level).
    unevenness   -- std(blur(Y)) / mean(blur(Y)). Gaano ka-uneven ang
                    ilaw sa image. Dapat BUMABA sa new version.
    clip_pct     -- % ng pixels na 255 (blown out).
    chroma_shift -- avg change sa LAB a*,b* vs original. Dapat maliit
                    (ibig sabihin hindi nagagalaw ang undertone).
"""

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

IMAGES_ROOT = PROJECT_ROOT / "data" / "processed" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ssr_comparison"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SAMPLES_PER_FOLDER = 2
SEED = 42
PANEL_HEIGHT = 320          # display height ng bawat panel

SIGMA = 30.0
EPSILON = 1.0
STRENGTH = 1.0
GAIN_LIMITS = (0.5, 2.0)


# ---------------------------------------------------------- SSR versions --

def ssr_old(image, sigma=SIGMA, epsilon=EPSILON):
    """Current version sa ssr script (per-channel, I * I/L)."""
    img = image.astype(np.float64) + epsilon
    corrected = np.empty_like(img)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        illumination = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
        correction = np.exp(np.log(channel) - np.log(illumination))
        m = correction.mean()
        if m > 1e-6:
            correction /= m
        correction = np.clip(correction, 0.5, 2.0)
        corrected[:, :, c] = channel * correction
    return np.clip(corrected, 0, 255).astype(np.uint8)


def ssr_new(image, sigma=SIGMA, epsilon=EPSILON,
            strength=STRENGTH, gain_limits=GAIN_LIMITS):
    """Proposed version (luminance-based, I * target/L). Returns (img, gain)."""
    img = image.astype(np.float64) + epsilon
    lum = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
    illumination = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma, sigmaY=sigma)
    retinex = np.log(lum) - np.log(illumination)
    target = illumination.mean()
    gain = (np.exp(retinex) * target) / lum
    gain = np.clip(gain ** strength, *gain_limits)
    gain *= lum.mean() / (lum * gain).mean()
    corrected = img * gain[:, :, None]
    out = np.clip(corrected - epsilon, 0, 255).astype(np.uint8)
    return out, gain


# --------------------------------------------------------------- metrics --

def luminance(img):
    f = img.astype(np.float64)
    return 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]


def metrics(img, original_lab=None):
    y = luminance(img)
    blur = cv2.GaussianBlur(y, (0, 0), sigmaX=SIGMA, sigmaY=SIGMA)
    out = {
        "mean_Y": y.mean(),
        "unevenness": blur.std() / max(blur.mean(), 1e-6),
        "clip_pct": 100.0 * (img >= 255).any(axis=2).mean(),
    }
    if original_lab is not None:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        d = lab[:, :, 1:] - original_lab[:, :, 1:]
        out["chroma_shift"] = np.sqrt((d ** 2).sum(axis=2)).mean()
    return out


# ----------------------------------------------------------- visualizing --

def gain_to_color(gain, limits=GAIN_LIMITS):
    """Asul = pinadilim (gain<1), puti = walang galaw, pula = pinaliwanag."""
    lo, hi = np.log(limits[0]), np.log(limits[1])
    g = np.log(gain)
    t = np.where(g >= 0, g / hi, g / -lo)           # -1..1
    t = np.clip(t, -1, 1)
    h, w = gain.shape
    vis = np.full((h, w, 3), 255.0)
    pos, neg = np.clip(t, 0, 1), np.clip(-t, 0, 1)
    vis[:, :, 0] -= 255 * pos                        # less blue  -> red
    vis[:, :, 1] -= 255 * (pos + neg)                # less green -> both
    vis[:, :, 2] -= 255 * neg                        # less red   -> blue
    return np.clip(vis, 0, 255).astype(np.uint8)


def labeled(img, text):
    h, w = img.shape[:2]
    scale = PANEL_HEIGHT / h
    panel = cv2.resize(img, (max(1, int(w * scale)), PANEL_HEIGHT),
                       interpolation=cv2.INTER_AREA)
    bar = np.full((28, panel.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


# ---------------------------------------------------------------- samples --

def pick_samples():
    if len(sys.argv) > 1:
        return [Path(p).resolve() for p in sys.argv[1:]]

    if not IMAGES_ROOT.is_dir():
        raise SystemExit(f"ERROR: {IMAGES_ROOT} not found.")

    rng = random.Random(SEED)
    samples = []
    for folder in sorted(p for p in IMAGES_ROOT.iterdir() if p.is_dir()):
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        samples += rng.sample(files, min(SAMPLES_PER_FOLDER, len(files)))
    return samples


# ------------------------------------------------------------------ main --

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = pick_samples()
    print(f"Comparing {len(samples)} images -> {OUTPUT_DIR}")

    rows = []
    for path in samples:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skip (unreadable): {path}")
            continue

        old = ssr_old(img)
        new, gain = ssr_new(img)

        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        m_orig = metrics(img)
        m_old = metrics(old, orig_lab)
        m_new = metrics(new, orig_lab)

        grid = np.hstack([
            labeled(img, f"Original  Y={m_orig['mean_Y']:.0f}  U={m_orig['unevenness']:.3f}"),
            labeled(old, f"Old SSR   Y={m_old['mean_Y']:.0f}  U={m_old['unevenness']:.3f}"),
            labeled(new, f"New SSR   Y={m_new['mean_Y']:.0f}  U={m_new['unevenness']:.3f}"),
            labeled(gain_to_color(gain), "Gain: blue=darker red=brighter"),
        ])

        folder = path.parent.name
        out_path = OUTPUT_DIR / f"{folder}__{path.stem}.png"
        cv2.imwrite(str(out_path), grid)

        row = {"folder": folder, "file": path.name}
        for tag, m in (("orig", m_orig), ("old", m_old), ("new", m_new)):
            for k, v in m.items():
                row[f"{tag}_{k}"] = round(float(v), 4)
        rows.append(row)

    if not rows:
        raise SystemExit("No images processed.")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    print("\nAverage across samples:")
    print(f"{'':10}{'mean_Y':>9}{'uneven':>9}{'clip%':>8}{'chroma':>9}")
    for tag in ("orig", "old", "new"):
        chroma = df[f"{tag}_chroma_shift"].mean() if f"{tag}_chroma_shift" in df else 0.0
        print(f"{tag:10}{df[f'{tag}_mean_Y'].mean():9.1f}"
              f"{df[f'{tag}_unevenness'].mean():9.3f}"
              f"{df[f'{tag}_clip_pct'].mean():8.2f}"
              f"{chroma:9.2f}")

    print("\nTingnan kung:")
    print("  - new 'uneven' < orig  (napantay ang ilaw)")
    print("  - new 'mean_Y' ~= orig (hindi nagalaw ang skin-tone level)")
    print("  - new 'chroma' maliit  (hindi nagalaw ang undertone)")


if __name__ == "__main__":
    main()