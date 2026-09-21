import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = "/mnt/c/Users/Elizabeth/HueView-tool/ml_pipeline"
MANIFEST = os.path.join(REPO_ROOT, "data/processed/manifest.csv")
IMAGES_DIR = os.path.join(REPO_ROOT, "data/processed/images")
OUTPUT = os.path.join(REPO_ROOT, "results/ssr_sigma30_visual.png")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

df = pd.read_csv(MANIFEST)

picks = (
    df.sort_values("filename")
      .groupby("SCC_label")
      .first()
      .reset_index()
)

def compute_ssr(img_rgb, sigma=30):
    img_f = img_rgb.astype(np.float64) + 1.0

    img_ssr = (
        np.log(img_f)
        - np.log(cv2.GaussianBlur(img_f, (0, 0), sigma) + 1e-6)
    )

    for c in range(3):
        ch = img_ssr[..., c]
        img_ssr[..., c] = (
            (ch - ch.min())
            / (ch.max() - ch.min() + 1e-6)
            * 255
        )

    return np.clip(img_ssr, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, len(picks), figsize=(18, 6))

for i, (_, row) in enumerate(picks.iterrows()):
    filename = row["filename"]
    scc = row["SCC_label"]

    img_path = os.path.join(IMAGES_DIR, filename)

    print(f"Processing {scc}: {filename}")

    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read: {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ssr = compute_ssr(img, sigma=30)

    axes[0, i].imshow(img)
    axes[0, i].set_title(f"{scc} - Original")
    axes[0, i].axis("off")

    axes[1, i].imshow(ssr)
    axes[1, i].set_title(f"{scc} - SSR sigma=30")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close()

print()
print("===================================")
print("DONE!")
print(f"Saved to: {OUTPUT}")
print("===================================")