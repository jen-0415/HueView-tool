import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = "/mnt/c/Users/Elizabeth/HueView-tool/ml_pipeline"
MANIFEST = os.path.join(REPO_ROOT, "data/processed/manifest.csv")
IMAGES_DIR = os.path.join(REPO_ROOT, "data/processed/images")
OUTPUT = os.path.join(REPO_ROOT, "results/ssr_corrected_sigma80_vs_30(1)).png")

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


def ssr_corrected(img_rgb, sigma=30, strength=1.0):
    """
    Apply SSR as an illumination correction to the
    original image instead of displaying the raw SSR output.

    strength:
        0.0 = original image
        1.0 = full correction
    """

    img = img_rgb.astype(np.float32) / 255.0

    # Small value to avoid log(0)
    img_safe = img + 1e-6

    # Estimate illumination
    blurred = cv2.GaussianBlur(
        img_safe,
        (0, 0),
        sigma
    )

    # SSR / reflectance estimate
    ssr = np.log(img_safe) - np.log(blurred + 1e-6)

    # Convert SSR into a correction factor.
    # Center around zero so we don't simply brighten
    # the entire image.
    correction = np.exp(ssr)

    # Normalize correction around 1
    correction_mean = np.mean(correction, axis=(0, 1), keepdims=True)
    correction = correction / (correction_mean + 1e-6)

    # Limit extreme correction values
    correction = np.clip(correction, 1.0, 2.0)

    # Blend correction with original image.
    corrected = img * (
        (1.0 - strength)
        + strength * correction
    )

    corrected = np.clip(corrected, 0, 1)

    return (corrected * 255).astype(np.uint8)


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
    corrected80 = ssr_corrected(
        img,
        sigma=80,
        strength=1.0
    )

    axes[1, i].imshow(corrected80)
    axes[1, i].set_title(
        f"{scc}\nSSR corrected σ=80"
    )
    axes[1, i].axis("off")

    # -------------------------
    # SSR sigma = 30
    # -------------------------
    corrected30 = ssr_corrected(
        img,
        sigma=30,
        strength=1.0
    )

    axes[2, i].imshow(corrected30)
    axes[2, i].set_title(
        f"{scc}\nSSR corrected σ=30"
    )
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