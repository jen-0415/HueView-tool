import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = "/mnt/c/Users/Elizabeth/HueView-tool/ml_pipeline"
MANIFEST = os.path.join(REPO_ROOT, "data/processed/manifest.csv")
IMAGES_DIR = os.path.join(REPO_ROOT, "data/processed/images")
OUTPUT = os.path.join(REPO_ROOT, "results/ssr_sigma80_vs_30.png")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

df = pd.read_csv(MANIFEST)

# Get one readable image per SCC
picks = []

for scc in sorted(df["SCC_label"].dropna().unique()):
    subset = df[df["SCC_label"] == scc]

    for _, row in subset.iterrows():
        img_path = os.path.join(IMAGES_DIR, row["filename"])

        if os.path.exists(img_path):
            img = cv2.imread(img_path)

            if img is not None:
                picks.append(row)
                break

print(f"Selected {len(picks)} images.")


def compute_ssr(img_rgb, sigma=30):
    """
    SSR with conservative global normalization.
    No separate normalization per RGB channel.
    """

    img_f = img_rgb.astype(np.float64) + 1.0

    blurred = cv2.GaussianBlur(
        img_f,
        (0, 0),
        sigma
    )

    ssr = np.log(img_f) - np.log(blurred + 1e-6)

    # Use robust percentiles instead of min/max.
    # This prevents extreme pixels from stretching
    # the whole image.
    low = np.percentile(ssr, 1)
    high = np.percentile(ssr, 99)

    ssr = np.clip(ssr, low, high)

    ssr = (
        (ssr - low)
        / (high - low + 1e-6)
        * 255
    )

    return ssr.astype(np.uint8)


fig, axes = plt.subplots(
    3,
    len(picks),
    figsize=(18, 9)
)

if len(picks) == 1:
    axes = axes.reshape(3, 1)


for i, row in enumerate(picks):

    filename = row["filename"]
    scc = row["SCC_label"]

    img_path = os.path.join(IMAGES_DIR, filename)

    print(f"Processing {scc}: {filename}")

    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read: {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -------------------------
    # Original
    # -------------------------
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"{scc}\nOriginal")
    axes[0, i].axis("off")

    # -------------------------
    # SSR sigma = 80
    # -------------------------
    ssr80 = compute_ssr(img, sigma=80)

    axes[1, i].imshow(ssr80)
    axes[1, i].set_title(f"{scc}\nSSR σ=80")
    axes[1, i].axis("off")

    # -------------------------
    # SSR sigma = 30
    # -------------------------
    ssr30 = compute_ssr(img, sigma=30)

    axes[2, i].imshow(ssr30)
    axes[2, i].set_title(f"{scc}\nSSR σ=30")
    axes[2, i].axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print()
print("===================================")
print("DONE!")
print(f"Saved to:")
print(OUTPUT)
print("===================================")