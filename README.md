# HueView

A hybrid CNN and Single-Scale Retinex approach with CIELAB regional facial analysis for skin tone and undertone classification.

Undergraduate thesis, BS Computer Science — College of Computer and Information Sciences, Polytechnic University of the Philippines.

Aldover, J. · Dela Cruz, C. · Flororita, E. · Lumabi, J.

The repository holds two applications:

- **`ml_pipeline/`** — the Python side: dataset preprocessing, the Baseline and HueView models, evaluation, the single-image inference pipeline, and the FastAPI service.
- **`frontend/`** — a React + Vite web app that uploads a face photo to the API and shows the Baseline-vs-HueView comparison.

---

## Repository structure

```text
HueView-tool/
├── README.md
├── requirements.txt              # Python dependencies to install (unpinned; includes the API server)
├── download_landmarker.py        # one-time: fetches ml_pipeline/face_landmarker.task
├── merge_mst.py                  # one-time, DESTRUCTIVE: flattens the image batch folders into MST-* (see below)
├── check_split.py                # SCC class balance across train/val/test
├── peek_baseline.py              # print one Baseline prediction (edit the hardcoded image path first)
├── peek_hueview.py               # print one full dual-model prediction (edit the image path first)
├── verify_phase10.py             # pre-training readiness check (checks the older *_verified.csv splits)
├── verify_names.py               # one-off SSR old-vs-new check (hardcoded path)
├── results/                      # stray copy of phase7_routing_log.csv (the pipeline writes to ml_pipeline/results/)
│
├── frontend/
│   ├── .env.example              # copy to .env: VITE_USE_MOCK, VITE_API_BASE
│   ├── package.json              # npm scripts: dev, build, lint, preview
│   └── src/
│       ├── api.js                # every call to the backend (and the mock path)
│       ├── useAnalysis.js        # upload -> confirm -> analyze (SSR preview) -> results
│       ├── constants.js          # SCC labels/colours, undertone rule text
│       ├── mockData.js           # used only when VITE_USE_MOCK=true
│       ├── components/           # model cards, colour values, undertone, regional segmentation
│       └── screens/              # Upload, Confirm, Analyzing, Results
│
└── ml_pipeline/                  # Python working root -- most scripts expect to run from here
    ├── requirements.txt          # pinned pip freeze of the training env (UTF-16; no API packages)
    ├── face_landmarker.task      # local only (gitignored): MediaPipe face mesh model
    ├── configs/                  # frozen inference-time values
    │   ├── hsv_skin_thresholds.json    # Phase 7.4 HSV skin filter
    │   ├── illumination_thresholds.json# Phase 3 k-means boundaries
    │   ├── scc_labels.json             # SCC-1..6 display names
    │   └── train_config.json           # HueView training hyperparameters
    ├── data/processed/           # split CSVs and metadata (committed); images gitignored
    ├── models/                   # local only (gitignored): trained weights
    ├── results/                  # evaluation outputs, training logs, confusion matrices
    ├── notebooks/                # phase3_kmeans_illumination_binning.ipynb
    ├── docs/                     # implementation notes and adviser drafts
    ├── tests/test_phase7_smoke.py
    ├── *.py                      # one-off diagnostics (check_*.py, split_leakage.py, verify_*.py, ...)
    ├── ssr_compare.py            # old-vs-new SSR side-by-side tool
    └── src/
        ├── preprocessing/        # Phases 1-5: MTCNN crop, luminance, illumination clusters, labels, splits
        ├── baseline/             # Phase 6: EfficientNetB0 + global RGB mean
        ├── hueview/              # Phases 7-12: SSR, landmarks, regions, HSV filter, CIELAB, undertone, train, evaluate
        ├── inference/            # Phase 14: single-image pipeline used by the API
        └── api/                  # Phase 15: FastAPI service
```

---

## Setup

### Python

Python 3.12 is recommended (3.11 also works). Do not use 3.13+: MediaPipe has no build for 3.13+, and TensorFlow has none for 3.14 — on 3.14 pip reports `Could not find a version that satisfies the requirement tensorflow`, which looks like a network problem but isn't.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
python download_landmarker.py          # one-time, writes ml_pipeline/face_landmarker.task
```

Install the root `requirements.txt`. `ml_pipeline/requirements.txt` is a pinned freeze of the training environment for reproducing exact versions; it lacks `fastapi`, `uvicorn`, `python-multipart` and `facenet-pytorch`, so the API and preprocessing will not run from it alone.

**Training does not run locally.** TensorFlow dropped native Windows GPU support at 2.11; training runs on Colab. Preprocessing, inference, evaluation scripts and diagnostics run locally.

### Model weights

Weights are gitignored and shared via Google Drive. Place them in `ml_pipeline/models/`:

| File | Used by |
|---|---|
| `baseline_effnet_final.h5` | the API (`BASELINE_WEIGHTS_FILENAME` in `src/inference/models.py`) |
| `hueview_<region>_<tag>.h5` + `lab_scaler_<region>_<tag>.pkl` for forehead, left_cheek, right_cheek, nose_bridge, jawline, full_face | the API (`HUEVIEW_SUFFIXES = ("_final1", "_final")`: each region loads `_final1` if both files exist, otherwise `_final`) |

Currently the five regions load `_final1` and full_face falls back to `_final` (there is no `full_face_final1` yet). The UI's "ckpt" field shows which files were loaded.

`src/hueview/evaluate.py` loads `baseline_effnet.h5` and takes the HueView tag from `--run-tag` (default `final`). If a model's files are missing, the API still starts and returns that model in placeholder mode (no SCC).

---

## Running

### API

```powershell
cd ml_pipeline
uvicorn src.api.main:app --port 8000 --reload
```

Interactive docs: http://localhost:8000/docs. Models load once at startup (20-30 s). Without `--reload`, restart the server after changing Python code.

| Endpoint | Purpose |
|---|---|
| `POST /api/detect` | MTCNN face check, returns the 224 × 224 crop preview |
| `POST /api/analyze` | registers a job, returns `{job_id}` |
| `GET /api/analyze/{job_id}/events` | SSE stream: `stage` events, an `ssr` event (SSR intermediate images), then `result` |
| `GET /api/health` | which models are loaded |
| `GET /api/config` | SCC labels, region names, stage keys |

CORS allows only `http://localhost:5173` and `http://127.0.0.1:5173` (the Vite dev server).

### Frontend

```powershell
cd frontend
npm install
copy .env.example .env      # optional; defaults shown in the file
npm run dev                 # http://localhost:5173
npm run build               # production build into frontend/dist/
```

### Tests

```powershell
# from the repo root
python -m ml_pipeline.tests.test_phase7_smoke
```

### Working directory and imports

Most pipeline scripts use paths like `Path("data/processed")` relative to `ml_pipeline/`, so run them from there. `src/inference` and `src/hueview` locate `ml_pipeline/` themselves and work from anywhere. Modules that import `ml_pipeline.src.…` (most of `src/baseline`, `src/hueview`, the `peek_*.py` scripts and the tests) need the repo root on the import path: run them as modules from the repo root (`python -m ml_pipeline.src.hueview.train`), or set `$env:PYTHONPATH = ".."` when running from `ml_pipeline/`.

---

## How HueView classifies an image

1. MTCNN crop to 224 × 224 (same settings as Phase 2).
2. SSR illumination normalization (`src/hueview/ssr_normalization.py`, σ = 50; `images_ssr/` and the current models were made with an earlier SSR).
3. MediaPipe face mesh → five regions (forehead, cheeks, nose bridge, jawline) as landmark convex hulls.
4. HSV skin filter inside each region (`configs/hsv_skin_thresholds.json`, decided on the original crop).
5. Six models: one per region (skin-masked SSR patch + 3-D CIELAB) and one full-face (SSR face + 15-D CIELAB).
6. **Final SCC:** majority vote of the six predictions (five regions + full face); a tie on votes goes to the tied class with the highest mean softmax across the six (`majority_vote()` in `src/inference/hueview.py`). **Note:** `evaluate.py` still reports the full-face model alone (`--primary full_face`) until it is updated to the same rule.
7. Undertone: hue angle of each region's CIELAB mean (Warm > 65°, Neutral 55–65°, Cool < 55°), majority vote across the five regions.

The Baseline is EfficientNetB0 on the crop fused with the global RGB mean; its undertone uses the normalized blue ratio (Cool > 0.285, Neutral 0.275–0.285, Warm < 0.275).

### Known open issues

- The HueView weights were trained on SSR images from the previous `apply_ssr()` (`data/processed/images_ssr/`); inference uses the current one. Retrain or revert before trusting HueView's numbers.
- The undertone's CIELAB comes from the original crop in the app and `run_phase9_batch.py`, but from the SSR patches in `evaluate.py`.
- The HSV filter does not reliably remove eyebrows or bangs from the forehead region.

---

## The image folders

`ml_pipeline/data/processed/images/` is gitignored and comes from Google Drive. The dataset (43,221 images) was assembled in batches, each written to its own folder: `processed/`, `c1_processed/` (the 7-26 batch), `c2_processed/` and `v5_processed/`.

When a later batch reused a filename, the manifest recorded the collision with a `(n)` marker while the file kept its original name inside its own batch folder. **The suffix identifies the folder, not a duplicate.**

```
MST-8/foo.jpg        ->  processed/MST-8/foo.jpg
MST-8/foo (2).jpg    ->  c2_processed/MST-8/foo.jpg      (a different image)
MST-2/bar.png        ->  v5_processed/MST-2/bar.bmp      (v5 batch is .bmp)
```

This was verified empirically: each manifest row stores the `mean_y` of the image its labels came from, and `src/baseline/test_suffix_hypothesis.py` recomputes it from every candidate file (no suffix → `processed` 247/250; `(2)` → `c2_processed` 250/250). `src/baseline/resolve_manifest.py` verifies every row and writes `resolved_manifest.csv`, which the Baseline reads instead of re-deriving paths. Stripping the suffix would pair roughly 28% of the dataset with labels from a different image — silently.

`merge_mst.py` moves every batch folder into flat `images/MST-1 … MST-10` folders, renaming collisions to `name_1.jpg`. It cannot be undone and breaks the mapping above. `resolved_manifest.csv`, `src/baseline/path_resolver.py` and the Baseline part of `evaluate.py` expect batch folders; `landmark_extraction.py` and `ssr_normalization.py` handle the merged layout. Check which layout your copy has before running a phase.

### Splits

| Split set | train / val / test | Used by |
|---|---|---|
| `train.csv` / `val.csv` / `test.csv` (= `baseline_rgb_*.csv`) | 30,074 / 8,618 / 4,529 | Baseline (frozen Phase 5 split — do not regenerate) |
| `*_hueview_usable.csv` | 29,912 / 8,584 / 4,488 | HueView training and evaluation |
| `*_verified.csv` | 19,610 / 5,634 / 2,994 | older scripts only (`verify_phase10.py`) |

`src/baseline/check_dataset.py` confirms no image appears in more than one of train/val/test.

---

## Notes for whoever picks this up

**Do not normalize images before the CNNs.** Keras' EfficientNet expects pixels in `[0, 255]` and rescales inside the model graph. The Baseline's RGB-mean input *is* scaled to `[0, 1]` — a different input with a different convention.

**SCC label order is shared across several places:** `frontend/src/constants.js`, `ml_pipeline/configs/scc_labels.json`, `SCC_CLASS_ORDER` in `src/inference/baseline.py` and `hueview.py`, and the models' output order. A mismatch mislabels every prediction without erroring.

**Check the undertone distribution before it reaches the results chapter.** The Baseline's Neutral band is only 0.01 wide; if one class dominates, report it plainly.

**Trained weights are gitignored** (`models/`, `*.h5`, `*.keras`, `*.pkl`). Agree on one Drive location.
