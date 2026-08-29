# HueView

A hybrid CNN and Single-Scale Retinex approach with CIELAB regional facial analysis for skin tone and undertone classification.

Undergraduate thesis, BS Computer Science — College of Computer and Information Sciences, Polytechnic University of the Philippines.

Aldover, J. · Dela Cruz, C. · Flororita, E. · Lumabi, J.

---

## ⚠️ Python 3.12 required

Do not use 3.13 or 3.14. Two hard blocks:

- **MediaPipe** has no build for Python 3.13+ (needed for Phase 7 landmark extraction)
- **TensorFlow** has no build for Python 3.14

On 3.14 you get `ERROR: Could not find a version that satisfies the requirement tensorflow (from versions: none)`, which looks like a network problem but isn't. Python 3.12.10 is the last 3.12 with a Windows installer — later 3.12.x are source-only.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify:

```powershell
python -c "import tensorflow as tf, mediapipe; print(tf.__version__, mediapipe.__version__)"
```

**Training does not run locally.** TensorFlow dropped native Windows GPU support at 2.11, so there is no GPU path on Windows regardless of your hardware. Phase 6.4 onward runs on Colab. Everything else — preprocessing, feature extraction, the undertone rule, all diagnostics — runs fine locally.

---

## Structure

```
HueView-tool/
├── data/processed/          # gitignored except CSV/JSON metadata
│   ├── images/              # four batch folders (see below)
│   ├── manifest.csv         # filename, MST/SCC labels, mean_y, illumination
│   ├── train.csv            # frozen Phase 5 split — do not regenerate
│   ├── val.csv
│   ├── test.csv
│   └── resolved_manifest.csv  # verified filename -> file mapping
├── src/
│   ├── preprocessing/       # Phases 1-5
│   └── baseline/            # Phase 6
└── requirements.txt
```

### `src/baseline/`

**Pipeline**

| File | Phase | Purpose |
|---|---|---|
| `path_resolver.py` | — | Resolves manifest filenames to files across the four image roots |
| `resolve_manifest.py` | — | One-time verified pass producing `resolved_manifest.csv` |
| `global_rgb_features.py` | 6.1 | Per-channel RGB means → 3-value feature vector |
| `model.py` | 6.2 | EfficientNetB0 + RGB fusion → 6-class softmax |
| `undertone.py` | 6.3 | Rule-based Warm/Neutral/Cool classification |

**Diagnostics** (one-off; kept as evidence for how the dataset was resolved)

| File | Question it answers |
|---|---|
| `check_dataset.py` | Duplicate collisions, cross-split leakage, root conflicts |
| `characterize_roots.py` | How versions differ across folders |
| `verify_root_choice.py` | Which root each row's labels came from |
| `test_suffix_hypothesis.py` | What the `(n)` filename suffix means |

---

## The four image folders

`data/processed/images/` contains `processed/`, `c2_processed/`, `processed - 7-26/`, and `v5_processed/`. This is not a mistake, and understanding it matters before touching anything image-related.

The dataset was assembled in batches over time, growing to 43,221 images. When a later batch contained a filename an earlier batch already used, the manifest recorded the collision with a `(n)` marker while the file itself was written into that batch's own folder under the original name. **The suffix identifies which folder a row refers to. It is not a duplicate marker.**

```
MST-8/foo.jpg        ->  processed/MST-8/foo.jpg
MST-8/foo (2).jpg    ->  c2_processed/MST-8/foo.jpg      (a different image)
MST-2/bar.png        ->  v5_processed/MST-2/bar.bmp      (v5 batch is .bmp)
```

Verified empirically rather than assumed. Each manifest row stores the `mean_y` of the image its labels were computed from, so `test_suffix_hypothesis.py` recomputes luminance from every candidate file and checks which reproduces it. Across 250 rows per group:

| Manifest row | Matching root |
|---|---|
| no suffix | `processed` — 247/250 (99%) |
| `(2)` suffix | `c2_processed` — 250/250 (100%) |

Rule-based routing handles 43,203 of 43,221 rows correctly. The remainder — 18 rows using `(1)` — are inconsistent, so `resolve_manifest.py` verifies **every** row against its stored `mean_y` and writes the result to `resolved_manifest.csv`. All later phases read that table instead of re-deriving paths.

### Why this matters

Stripping the suffix and serving the `processed` version would pair roughly 28% of the dataset with labels computed from a different rendering of that image. Nothing errors. The numbers just come out wrong.

`resolved_manifest.csv` is committed deliberately. It is part of what makes the Baseline-vs-HueView comparison reproducible, and it guarantees both models read identical bytes for identical rows.

### Verified clean

`check_dataset.py` confirms **no cross-split leakage** — no image appears in more than one of train/val/test. The Phase 5 splits are sound and must not be regenerated; the methodology requires both models reuse them unchanged.

---

## Phase 6 status

| | State |
|---|---|
| 6.1 Global RGB extraction | Code complete; needs `resolved_manifest.csv` |
| 6.2 Classification branch | Complete and verified |
| 6.3 Undertone rule | Complete and verified |
| 6.4 Training | Pending — needs 6.1 output, runs on Colab |
| 6.5 Evaluation | Pending |

### Notes for whoever picks this up

**Do not normalize images before the CNN.** Keras' EfficientNet expects pixels in `[0, 255]` and rescales inside the model graph. Dividing by 255 first normalizes twice and quietly costs accuracy. The 6.1 RGB features *are* scaled to `[0, 1]`, but those feed the other branch — the two are different inputs with different conventions.

**The RGB branch is outnumbered 1280-to-3.** Straight concatenation is what the methodology specifies, so that's the default, but the CNN branch dominates the gradient almost entirely. `model.py` exposes `rgb_projection_dim` if this is ever revisited — with the caveat that HueView's fusion needs the same treatment, or the comparison stops being like-for-like.

**Check the undertone distribution before it reaches the results chapter.** The Neutral band spans only 0.01 (b_ratio 0.275–0.285) and skin tones cluster tightly. If one class dominates, that is a finding to report plainly, not three categories to present as equally exercised. `undertone.py` prints the distribution and flags any class above 90%.

**Trained weights are gitignored** (`models/`, `*.h5`, `*.keras`). They live on Google Drive. Agree on a location before someone's laptop dies.

---

## Running things

Always run from the repo root — paths inside the scripts are relative to it.

```powershell
# One-time, after any manifest change (slow: ~1-2 hours over 43k rows)
python src/baseline/resolve_manifest.py

# Confirm dataset integrity
python src/baseline/check_dataset.py

# Phase 6.1
python src/baseline/global_rgb_features.py

# Phase 6.2 — architecture check
python src/baseline/model.py

# Phase 6.3
python src/baseline/undertone.py
```
