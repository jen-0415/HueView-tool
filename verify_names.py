import random
from pathlib import Path
import cv2
import numpy as np

BASE = Path(r"C:\Users\Elizabeth\HueView-tool\ml_pipeline\data\processed")
IMG, OLD = BASE / "images", BASE / "images_ssr_old"


def ssr_old(image, sigma=30.0, eps=1.0):
    img = image.astype(np.float64) + eps
    out = np.empty_like(img)
    for c in range(3):
        ch = img[:, :, c]
        L = cv2.GaussianBlur(ch, (0, 0), sigmaX=sigma, sigmaY=sigma)
        corr = np.exp(np.log(ch) - np.log(L))
        corr /= max(corr.mean(), 1e-6)
        out[:, :, c] = ch * np.clip(corr, 0.5, 2.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return np.nan
    return float(np.abs(a.astype(float) - b.astype(float)).mean())


names = [p.relative_to(OLD) for p in OLD.rglob("* (2).jpg")]
print(f"'(2)' files sa images_ssr_old: {len(names)}")
random.seed(0)
sample = random.sample(names, min(50, len(names)))

d1s, d0s = [], []
for rel in sample:
    ref = cv2.imread(str(OLD / rel))
    p1 = IMG / rel.parent / rel.name.replace(" (2).jpg", "_1.jpg")
    p0 = IMG / rel.parent / rel.name.replace(" (2).jpg", ".jpg")
    i1, i0 = cv2.imread(str(p1)), cv2.imread(str(p0))
    d1s.append(diff(ssr_old(i1), ref) if i1 is not None else np.nan)
    d0s.append(diff(ssr_old(i0), ref) if i0 is not None else np.nan)

d1s, d0s = np.array(d1s), np.array(d0s)
print(f"Median diff vs x_1.jpg : {np.nanmedian(d1s):.2f}")
print(f"Median diff vs x.jpg   : {np.nanmedian(d0s):.2f}")
print(f"x_1 mas malapit kaysa x : {int(np.sum(d1s < d0s))} / {len(sample)}")
print(f"x_1 halos pareho (<5)   : {int(np.sum(d1s < 5))} / {len(sample)}")
print(f"Walang x.jpg (NaN)      : {int(np.sum(np.isnan(d0s)))} / {len(sample)}")
print(f"x.jpg halos pareho (<5) : {int(np.sum(d0s < 5))} / {len(sample)}")