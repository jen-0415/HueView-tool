# HueView diagnostic checks — COMPLETE VERSION (fixed for local repo)
# Run in your WSL environment from the repo root

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from skimage.color import rgb2lab
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ========== CONFIG ==========
REPO_ROOT = "/mnt/c/Users/Elizabeth/HueView-tool/ml_pipeline"
MANIFEST = os.path.join(REPO_ROOT, "data/processed/manifest.csv")
IMAGES_DIR = os.path.join(REPO_ROOT, "data/processed/images")
RESULTS_CSV = os.path.join(REPO_ROOT, "results/classification_results_record.csv")
PHASE7_STATS = os.path.join(REPO_ROOT, "data/processed/phase7_4_full_coverage_stats.csv")

N_PER_CLASS = 150
OUTPUT_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("HueView Diagnostic: SSR & HSV Impact Analysis")
print("=" * 70)

# %% CELL 0 — Load and sample data
print("\n[CELL 0] Loading manifest and sampling...")
try:
    df = pd.read_csv(MANIFEST)
    print(f"✓ Loaded manifest: {len(df)} images")
    print(f"  Columns: {list(df.columns)}")
except FileNotFoundError as e:
    print(f"✗ Manifest not found: {e}")
    exit(1)

print(f"\nSample row:\n{df.iloc[0]}")
print(f"\nSCC label distribution:\n{df['SCC_label'].value_counts().sort_index()}")

# Sample for diagnostics
sample_list = []
for scc_label in sorted(df['SCC_label'].unique()):
    group = df[df['SCC_label'] == scc_label]
    n_sample = min(len(group), N_PER_CLASS)
    sample_list.append(group.sample(n=n_sample, random_state=42))
sample = pd.concat(sample_list, ignore_index=True)

print(f"\nSampled {len(sample)} images ({N_PER_CLASS} per class max)")
print(f"Sampled SCC distribution:\n{sample['SCC_label'].value_counts().sort_index()}")

# %% CELL 1 — Does skin-tone signal survive SSR?
print("\n" + "=" * 70)
print("[CELL 1] SSR Impact: Does L* survive illumination normalization?")
print("=" * 70)

def load_rgb(path):
    """Load image from path, return RGB array or None."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_skin_mask(img_rgb, margin=20):
    """Rough skin mask: central region."""
    mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    return mask

def compute_ssr(img_rgb, sigma=80):
    """Compute SSR-normalized image."""
    img_f = img_rgb.astype(np.float64) + 1.0
    img_ssr = np.log(img_f) - np.log(cv2.GaussianBlur(img_f, (0, 0), sigma) + 1e-6)
    # Min-max rescale per channel
    for c in range(3):
        ch = img_ssr[..., c]
        img_ssr[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6) * 255
    return np.clip(img_ssr, 0, 255).astype(np.uint8)

rows = []
failed = 0
for idx, (_, r) in enumerate(sample.iterrows()):
    filename = r['filename']
    img_path = os.path.join(IMAGES_DIR, filename)
    
    if not os.path.exists(img_path):
        failed += 1
        continue
    
    raw = load_rgb(img_path)
    if raw is None:
        failed += 1
        continue
    
    m = get_skin_mask(raw)
    
    # Lab from raw
    L0, a0, b0 = rgb2lab(raw / 255.0)[m].mean(0)
    
    # Compute SSR
    img_ssr = compute_ssr(raw, sigma=80)
    L1, a1, b1 = rgb2lab(img_ssr / 255.0)[m].mean(0)
    
    rows.append(dict(
        filename=filename,
        SCC=r['SCC_label'],
        L_raw=L0, a_raw=a0, b_raw=b0,
        L_ssr=L1, a_ssr=a1, b_ssr=b1
    ))
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(sample)}")

lab = pd.DataFrame(rows)
print(f"\n✓ Processed {len(lab)} images ({failed} failed)")

print("\n--- Mean CIELAB values by SCC class ---")
summary = lab.groupby("SCC")[["L_raw", "L_ssr", "a_raw", "a_ssr", "b_raw", "b_ssr"]].mean().round(1)
print(summary)

# Convert SCC labels to numeric for correlation
lab['SCC_numeric'] = lab['SCC'].str.extract('(\d+)').astype(int)

r_raw = spearmanr(lab.SCC_numeric, lab.L_raw)[0]
r_ssr = spearmanr(lab.SCC_numeric, lab.L_ssr)[0]

print(f"\n--- Correlation between SCC class and L* ---")
print(f"Raw images:        ρ = {r_raw:7.3f}  (ideal: >0.7)")
print(f"After SSR:         ρ = {r_ssr:7.3f}  (ideal: >0.7)")
print(f"Signal loss:       {abs(r_raw) - abs(r_ssr):7.3f}  (should be ~0)")

if abs(r_raw) > 0.5 and abs(r_ssr) < 0.3:
    print("\n⚠️  CRITICAL: SSR is destroying or inverting the tone signal!")
    if r_ssr < 0:
        print("   → L* is INVERTED after SSR (darkest skin now brightest)")
elif abs(r_ssr) < abs(r_raw) - 0.2:
    print("\n⚠️  WARNING: SSR significantly reduces tone correlation")
else:
    print("\n✓ SSR preserves tone signal reasonably well")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
lab.boxplot("L_raw", by="SCC", ax=ax[0])
ax[0].set_title("Mean L* — Raw Images")
ax[0].set_xlabel("SCC Class")
ax[0].set_ylabel("L*")
ax[0].grid(True, alpha=0.3)

lab.boxplot("L_ssr", by="SCC", ax=ax[1])
ax[1].set_title("Mean L* — After SSR (σ=80, per-image min-max)")
ax[1].set_xlabel("SCC Class")
ax[1].set_ylabel("L*")
ax[1].grid(True, alpha=0.3)

plt.suptitle("")
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "diagnostic_01_ssr_impact.png")
plt.savefig(plot_path, dpi=100, bbox_inches="tight")
print(f"\n✓ Plot saved: {plot_path}")
plt.close()

# %% CELL 1b — Cheap ablation: color-only classification
print("\n" + "=" * 70)
print("[CELL 1b] Ablation: Can 3 CIELAB numbers predict SCC?")
print("=" * 70)
print("(If raw >> SSR, SSR is removing/corrupting classification signal)\n")

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))

results_ablation = {}
for name, cols in [("Raw Lab", ["L_raw", "a_raw", "b_raw"]),
                   ("SSR Lab", ["L_ssr", "a_ssr", "b_ssr"])]:
    try:
        acc = cross_val_score(clf, lab[cols], lab.SCC_numeric, cv=5, scoring="balanced_accuracy")
        print(f"{name:15s}: {acc.mean():.3f} ± {acc.std():.3f}  (chance = {1/6:.3f})")
        results_ablation[name] = acc.mean()
    except Exception as e:
        print(f"{name:15s}: Error — {e}")

if "Raw Lab" in results_ablation and "SSR Lab" in results_ablation:
    diff = results_ablation["Raw Lab"] - results_ablation["SSR Lab"]
    print(f"\nDifference (Raw - SSR):  {diff:.3f}")
    if diff > 0.15:
        print("⚠️  SSR significantly removes discriminative color information")

# %% CELL 2 — Visual: raw vs SSR, one face per SCC
print("\n" + "=" * 70)
print("[CELL 2] Visual Check: Raw vs SSR (one image per SCC)")
print("=" * 70)

picks = sample.groupby("SCC_label").head(1).sort_values("SCC_label")
fig, ax = plt.subplots(2, len(picks), figsize=(2.8 * len(picks), 5))

for i, (_, r) in enumerate(picks.iterrows()):
    filename = r['filename']
    img_path = os.path.join(IMAGES_DIR, filename)
    
    if not os.path.exists(img_path):
        continue
    
    raw = load_rgb(img_path)
    if raw is None:
        continue
    
    img_ssr = compute_ssr(raw, sigma=80)
    
    ax[0, i].imshow(raw)
    ax[0, i].set_title(f"SCC-{r['SCC_label']}\nRaw")
    ax[0, i].axis("off")
    
    ax[1, i].imshow(img_ssr)
    ax[1, i].set_title("After SSR")
    ax[1, i].axis("off")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "diagnostic_02_visual_comparison.png")
plt.savefig(plot_path, dpi=100, bbox_inches="tight")
print(f"✓ Visual comparison saved: {plot_path}")
plt.close()

# %% CELL 3 — Region coverage and HSV pixel loss
print("\n" + "=" * 70)
print("[CELL 3] Region Masks: Coverage & HSV Pixel Loss by SCC")
print("=" * 70)

if os.path.exists(PHASE7_STATS):
    phase7 = pd.read_csv(PHASE7_STATS)
    print(f"✓ Loaded Phase 7 stats: {len(phase7)} rows")
    print(f"  Columns: {list(phase7.columns)[:15]}")
    
    # Check if we have relevant columns
    if 'SCC_label' in phase7.columns and 'region' in phase7.columns:
        if 'valid_pixels' in phase7.columns:
            cov = phase7.groupby(['SCC_label', 'region'])['valid_pixels'].agg(['median', 'mean']).round(0)
            print("\nMedian valid skin pixels per region, by SCC:")
            print(cov)
            
            # Check for regions with very few pixels
            low_px = phase7[phase7['valid_pixels'] < 200]
            if len(low_px) > 0:
                print(f"\n⚠️  {len(low_px)} / {len(phase7)} patches have <200 valid pixels (unreliable)")
                by_scc = low_px.groupby('SCC_label').size()
                print("Breakdown by SCC:")
                print(by_scc)
                pct = 100 * len(low_px) / len(phase7)
                print(f"That's {pct:.1f}% of all patches")
            else:
                print(f"✓ All patches have ≥200 valid pixels")
        else:
            print("⚠️  'valid_pixels' column not found in phase7 stats")
    else:
        print("⚠️  Expected columns 'SCC_label' and 'region' not found")
else:
    print(f"⚠️  Phase 7 stats not found: {PHASE7_STATS}")

# %% CELL 6 — Where does HueView lose?
print("\n" + "=" * 70)
print("[CELL 6] Results Breakdown: Where HueView loses to Baseline")
print("=" * 70)

if os.path.exists(RESULTS_CSV):
    res = pd.read_csv(RESULTS_CSV)
    print(f"✓ Loaded classification record: {len(res)} test images")
    print(f"  Columns: {list(res.columns)[:10]}")
    
    if 'baseline_predicted_SCC' in res.columns and 'hueview_predicted_SCC' in res.columns:
        res["base_ok"] = res.baseline_predicted_SCC == res.SCC_ground_truth
        res["hue_ok"]  = res.hueview_predicted_SCC  == res.SCC_ground_truth
        
        print(f"\n--- Overall Accuracy ---")
        base_acc = res.base_ok.mean()
        hue_acc = res.hue_ok.mean()
        print(f"Baseline:  {base_acc:.4f} ({int(res.base_ok.sum())}/{len(res)})")
        print(f"HueView:   {hue_acc:.4f} ({int(res.hue_ok.sum())}/{len(res)})")
        print(f"Diff (HV - Base):  {(hue_acc - base_acc):+.4f}")
        
        if 'illumination_label' in res.columns:
            print(f"\n--- Accuracy by Illumination ---")
            by_illum = res.groupby("illumination_label")[["base_ok", "hue_ok"]].agg(['sum', 'count'])
            by_illum['base_acc'] = res.groupby("illumination_label")['base_ok'].mean()
            by_illum['hue_acc'] = res.groupby("illumination_label")['hue_ok'].mean()
            by_illum['diff'] = by_illum['hue_acc'] - by_illum['base_acc']
            print(by_illum[['base_acc', 'hue_acc', 'diff']].round(4))
        
        if 'SCC_ground_truth' in res.columns:
            print(f"\n--- Accuracy by True SCC Class ---")
            by_class = res.groupby("SCC_ground_truth")[["base_ok", "hue_ok"]].agg(['sum', 'count'])
            by_class['base_acc'] = res.groupby("SCC_ground_truth")['base_ok'].mean()
            by_class['hue_acc'] = res.groupby("SCC_ground_truth")['hue_ok'].mean()
            by_class['diff'] = by_class['hue_acc'] - by_class['base_acc']
            print(by_class[['base_acc', 'hue_acc', 'diff']].round(4))
        
        print(f"\n--- HueView Prediction Distribution ---")
        print("(Collapse onto 1-2 classes = lost tone signal)")
        pred_dist = res.hueview_predicted_SCC.value_counts(normalize=True).sort_index().round(4)
        print(pred_dist)
        
        print(f"\n--- Discordant Pairs (where only one model is correct) ---")
        disc = res.assign(
            base_only = res.base_ok & ~res.hue_ok,
            hue_only = ~res.base_ok & res.hue_ok
        )
        base_only_ct = disc.base_only.sum()
        hue_only_ct = disc.hue_only.sum()
        both_correct = (res.base_ok & res.hue_ok).sum()
        both_wrong = (~res.base_ok & ~res.hue_ok).sum()
        
        print(f"Both correct:           {both_correct}")
        print(f"Both wrong:             {both_wrong}")
        print(f"Baseline-only correct:  {base_only_ct}")
        print(f"HueView-only correct:   {hue_only_ct}")
        
        if base_only_ct > hue_only_ct:
            print(f"\n⚠️  Baseline wins {base_only_ct - hue_only_ct} more cases — HueView losing signal")
        elif hue_only_ct > base_only_ct:
            print(f"\n✓ HueView wins {hue_only_ct - base_only_ct} more cases — HueView gaining")
        else:
            print(f"\n→ Tied on discordant pairs")
    else:
        print(f"⚠️  Expected columns not found. Got: {list(res.columns)}")
else:
    print(f"⚠️  Classification record not found: {RESULTS_CSV}")
    print("   Run Phase 11 (evaluation) first or check the path")

print("\n" + "=" * 70)
print("Diagnostic complete!")
print(f"Plots saved to: {OUTPUT_DIR}")
print("=" * 70)