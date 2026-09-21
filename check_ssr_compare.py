import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = "/mnt/c/Users/Elizabeth/HueView-tool/ml_pipeline"
MANIFEST = os.path.join(REPO_ROOT, "data/processed/manifest.csv")
IMAGES_DIR = os.path.join(REPO_ROOT, "data/processed/images")
OUTPUT = os.path.join(REPO_ROOT, "results/ssr_sigma30_compare.png")

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


def ssr_per_channel(img_rgb, sigma=30):
    """
    Current method:
    Each RGB channel is normalized separately.
    """
    img_f = img_rgb.astype(np.float64) + 1.0

    img_ssr = (
        np.log(img_f)
        - np.log(
            cv2.GaussianBlur(img_f, (0, 0), sigma) + 1e-6
        )
    )

    for c in range(3):
        ch = img_ssr[..., c]

        img_ssr[..., c] = (
            (ch - ch.min())
            / (ch.max() - ch.min() + 1e-6)
            * 255
        )

    return np.clip(img_ssr, 0, 255).astype(np.uint8)


def ssr_global(img_rgb, sigma=30):
    """
    Global/shared normalization:
    One min/max range is used for all RGB channels.
    """
    img_f = img_rgb.astype(np.float64) + 1.0

    img_ssr = (
        np.log(img_f)
        - np.log(
            cv2.GaussianBlur(img_f, (0, 0), sigma) + 1e-6
        )
    )

    # One shared min/max across R, G, and B
    min_val = img_ssr.min()
    max_val = img_ssr.max()

    img_ssr = (
        (img_ssr - min_val)
        / (max_val - min_val + 1e-6)
        * 255
    )

    return np.clip(img_ssr, 0, 255).astype(np.uint8)


# 3 rows:
# Original
# SSR sigma=30 per-channel
# SSR sigma=30 global
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

    # Original
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"{scc}\nOriginal")
    axes[0, i].axis("off")

    # Current per-channel normalization
    ssr_pc = ssr_per_channel(img, sigma=30)

    axes[1, i].imshow(ssr_pc)
    axes[1, i].set_title(f"{scc}\nSSR σ=30\nPer-channel")
    axes[1, i].axis("off")

    # New global normalization
    ssr_global_img = ssr_global(img, sigma=30)

    axes[2, i].imshow(ssr_global_img)
    axes[2, i].set_title(f"{scc}\nSSR σ=30\nGlobal")
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