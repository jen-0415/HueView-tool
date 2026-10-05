# HueView

A hybrid CNN and Single-Scale Retinex approach with CIELAB regional facial analysis for skin tone and undertone classification.

Undergraduate thesis, BS Computer Science — College of Computer and Information Sciences, Polytechnic University of the Philippines.

Aldover, J. · Dela Cruz, C. · Flororita, E. · Lumabi, J.

The repository holds two applications:

- **`ml_pipeline/`** — the Python side: dataset preprocessing, the Baseline and HueView models, the inference pipeline, and the FastAPI service that serves it.
- **`frontend/`** — a React + Vite web app that uploads a face image to the API and displays the Baseline-vs-HueView comparison.

---

## Repository structure

```text
HueView-tool/
├── README.md
├── requirements.txt              # Python dependencies (unpinned; includes the API server)
├── .gitignore
│
├── frontend/                     # React + Vite web app
│   ├── index.html
│   ├── package.json              # npm scripts: dev, build, lint, preview
│   ├── vite.config.js  tailwind.config.js  postcss.config.js  eslint.config.js
│   ├── .env                      # local only (gitignored): VITE_USE_MOCK, VITE_API_BASE
│   └── src/
│       ├── main.jsx  App.jsx  index.css
│       ├── api.js                # all calls to the backend
│       ├── useAnalysis.js        # upload -> analyze -> results state machine
│       ├── constants.js          # SCC labels / undertone rules (order must match the API)
│       ├── mockData.js           # used when VITE_USE_MOCK=true
│       ├── components/           # result cards, bars, tables
│       └── screens/              # Upload, Confirm, Analyzing, Results
│
└── ml_pipeline/                  # Python working root -- run things from here (see below)
    ├── requirements.txt          # pinned `pip freeze` of the training environment
    ├── face_landmarker.task      # local only (gitignored): MediaPipe model, see scripts/download_landmarker.py
    ├── configs/                  # frozen thresholds/labels loaded at inference time
    │   ├── hsv_skin_thresholds.json
    │   ├── illumination_thresholds.json
    │   ├── scc_labels.json
    │   └── train_config.json
    ├── data/processed/           # dataset metadata (CSV/JSON committed; images gitignored)
    ├── models/                   # local only (gitignored): trained .keras/.h5/.pkl weights
    ├── results/                  # training logs, histories, confusion matrices, metrics
    ├── notebooks/                # phase3_kmeans_illumination_binning.ipynb
    ├── docs/                     # implementation notes and adviser drafts
    │   └── figures/              # landmark/region reference images
    ├── tests/
    │   └── test_phase7_smoke.py
    ├── scripts/                  # standalone utilities (not imported by src/)
    │   ├── download_landmarker.py   # one-time: fetch face_landmarker.task
    │   ├── merge_mst.py             # one-time, DESTRUCTIVE: flattens image batch folders into MST-* (see below)
    │   ├── verify_phase10.py        # readiness check before training
    │   ├── check_split.py           # SCC class balance across train/val/test
    │   ├── peek_baseline.py         # print one Baseline prediction
    │   ├── peek_hueview.py          # print one full dual-model prediction
    │   └── diagnostics/             # one-off dataset investigations, kept as evidence
    └── src/
        ├── preprocessing/        # Phases 1-5: face crop, luminance, illumination clusters, labels, splits
        ├── baseline/             # Phase 6: EfficientNetB0 + global RGB baseline
        ├── hueview/              # Phases 7-12: SSR, landmarks, regions, HSV filter, CNN+CIELAB, undertone, train/eval
        ├── inference/            # Phase 14: single-image inference pipeline used by the API
        └── api/                  # Phase 15: FastAPI service
```

### `scripts/diagnostics/`

One-off investigations from resolving the dataset. They are not part of the pipeline; they are kept because they document how data problems were found.

| File | Purpose |
|---|---|
| `check.py` | List every CSV under `data/` |
| `check_alternate_roots.py` | Look for landmark-failure images under other image roots |
| `check_manifest_paths.py` | Compare manifest filenames against files on disk |
| `check_resolved_path_column.py` | Check `resolved_manifest.csv` paths exist on disk |
| `check_resolved_splits.py` | Cross-check split CSVs against the resolved manifest |
| `check_unreadable_files.py` | Find images OpenCV cannot read |
| `inspect_landmarks.py` | Print `landmarks.npy` shape; writes `landmark_preview.png` to the current directory |
| `landmark_failures_summary.py` | Count landmark failures by reason |
| `split_leakage.py` | Detect the same physical image in more than one split |
| `verify_fallback_matches.py` | Verify fallback-recovered paths against stored `mean_y` |
| `verify_splits.py` | Produce `*_verified.csv` (has a hardcoded `ROOT` — edit before running) |

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

There are two requirements files. The root `requirements.txt` is the one to install: unpinned, and it includes the API server (`fastapi`, `uvicorn`, `python-multipart`). `ml_pipeline/requirements.txt` is a pinned `pip freeze` of the training environment, useful for reproducing exact versions; it does not include the API packages.

**Training does not run locally.** TensorFlow dropped native Windows GPU support at 2.11, so there is no GPU path on Windows regardless of your hardware. Phase 6.4 onward runs on Colab. Everything else — preprocessing, feature extraction, the undertone rule, all diagnostics — runs fine locally.

---

## Running things

### Working directory and imports

**Run Python from `ml_pipeline/`.** Most scripts use paths like `Path("data/processed")`, `configs/`, `models/` and `results/` that are relative to `ml_pipeline/`. The `src/inference` and `src/hueview` modules instead locate `ml_pipeline/` themselves (by walking up to the folder containing `data/`), so they work from anywhere.

Two import styles exist in `src/`:

- **Relative / `src.…`** — `src/api` and `src/inference`. Run with `ml_pipeline/` as the working directory.
- **`ml_pipeline.src.…`** — most of `src/baseline` and `src/hueview`, plus `scripts/peek_*.py` and `tests/`. These need the **repo root** on the import path: either run them as modules from the repo root (`python -m ml_pipeline.…`), or, for the ones that also need `ml_pipeline/` as the working directory, set `PYTHONPATH` to the repo root:

  ```powershell
  cd ml_pipeline
  $env:PYTHONPATH = ".."
  python src/baseline/check_dataset.py
  ```

### API

```powershell
cd ml_pipeline
uvicorn src.api.main:app --port 8000
```

Interactive docs at http://localhost:8000/docs. Weights are loaded once at startup from `ml_pipeline/models/`; if they are missing the API starts in placeholder mode and logs a warning. `face_landmarker.task` must exist in `ml_pipeline/` (`python scripts/download_landmarker.py`).

Endpoints (`src/api/routes.py`): `POST /api/detect`, `POST /api/analyze`, `GET /api/analyze/{job_id}/events` (SSE), `GET /api/health`, `GET /api/config`.

### Frontend

```powershell
cd frontend
npm install
npm run dev       # http://localhost:5173
npm run build     # production build into frontend/dist/
```

Configure with `frontend/.env`:

| Variable | Default | Meaning |
|---|---|---|
| `VITE_USE_MOCK` | (unset = false) | `true` serves `mockData.js` and never calls the API |
| `VITE_API_BASE` | `http://localhost:8000` | API base URL |

The API's CORS allows only `http://localhost:5173` and `http://127.0.0.1:5173`.

### Tests

```powershell
# from the repo root
python -m ml_pipeline.tests.test_phase7_smoke
```

### Scripts

```powershell
cd ml_pipeline
python scripts/download_landmarker.py    # one-time
python scripts/check_split.py
$env:PYTHONIOENCODING = "utf-8"          # it prints check marks the default Windows console can't encode
python scripts/verify_phase10.py
python scripts/diagnostics/<name>.py

# peek_* import ml_pipeline.src, so run them as modules from the repo root.
# Edit the hardcoded image path inside each first.
cd ..
python -m ml_pipeline.scripts.peek_hueview
```

### Pipeline by phase

| Phase | Where | Entry points |
|---|---|---|
| 1–5 Preprocessing | `src/preprocessing/` | `mtcnn_filter.py`, `extract_luminance.py`, `cluster_illumination.py`, `assign_labels.py`, `add_scc_labels.py`, `generate_splits.py` |
| 6 Baseline | `src/baseline/` | `resolve_manifest.py`, `global_rgb_features.py`, `model.py`, `undertone.py`, `train.py`, `evaluate.py` |
| 7 SSR, landmarks, regions | `src/hueview/` | `ssr_normalization.py`, `landmark_extraction.py`, `segment_regions.py`, `run_phase7_4_batch.py`, `run_phase7_5_batch.py` |
| 8 Features + hybrid classifier | `src/hueview/` | `cnn_features.py`, `cielab_features.py`, `hybrid_classifier.py`, `run_phase8_features.py` |
| 9 Undertone | `src/hueview/` | `undertone.py`, `run_phase9_batch.py` |
| 10–12 Train / evaluate | `src/hueview/` | `python -m ml_pipeline.src.hueview.train` (see its docstring for flags), `evaluate.py` |
| 14 Inference | `src/inference/` | `pipeline.classify_image()`; `python -m src.inference.config` exports `configs/` |
| 15 API | `src/api/` | `main.py` |

Files named `qa_*`, `audit_*`, `check_*` and `visual_qa.py` in `src/hueview/`, and `check_dataset.py`, `characterize_roots.py`, `verify_root_choice.py`, `test_suffix_hypothesis.py`, `test_background_effect.py` in `src/baseline/`, are diagnostics for those phases. They are not unit tests.

`src/hueview/derive_regions.py`, `src/hueview/visualize_landmark.py` and `scripts/diagnostics/inspect_landmarks.py` write their PNGs to the current directory. The committed copies live in `ml_pipeline/docs/figures/`.

---

## The image folders

`ml_pipeline/data/processed/images/` is gitignored and comes from Google Drive. The dataset was assembled in batches over time, growing to 43,221 images, and each batch has its own folder: `processed/`, `c1_processed/` (the 7-26 batch), `c2_processed/` and `v5_processed/`. This is not a mistake, and understanding it matters before touching anything image-related.

When a later batch contained a filename an earlier batch already used, the manifest recorded the collision with a `(n)` marker while the file itself was written into that batch's own folder under the original name. **The suffix identifies which folder a row refers to. It is not a duplicate marker.**

```
MST-8/foo.jpg        ->  processed/MST-8/foo.jpg
MST-8/foo (2).jpg    ->  c2_processed/MST-8/foo.jpg      (a different image)
MST-2/bar.png        ->  v5_processed/MST-2/bar.bmp      (v5 batch is .bmp)
```

Verified empirically rather than assumed. Each manifest row stores the `mean_y` of the image its labels were computed from, so `src/baseline/test_suffix_hypothesis.py` recomputes luminance from every candidate file and checks which reproduces it. Across 250 rows per group:

| Manifest row | Matching root |
|---|---|
| no suffix | `processed` — 247/250 (99%) |
| `(2)` suffix | `c2_processed` — 250/250 (100%) |

Rule-based routing handles 43,203 of 43,221 rows correctly. The remainder — 18 rows using `(1)` — are inconsistent, so `src/baseline/resolve_manifest.py` verifies **every** row against its stored `mean_y` and writes the result to `resolved_manifest.csv`. The Baseline reads that table instead of re-deriving paths.

### Why this matters

Stripping the suffix and serving the `processed` version would pair roughly 28% of the dataset with labels computed from a different rendering of that image. Nothing errors. The numbers just come out wrong.

`resolved_manifest.csv` is committed deliberately. It is part of what makes the Baseline-vs-HueView comparison reproducible, and it guarantees both models read identical bytes for identical rows.

### Batch folders vs. merged `MST-*` layout

`scripts/merge_mst.py` moves every batch folder's contents into flat `images/MST-1 … MST-10` folders, renaming collisions to `name_1.jpg`. It cannot be undone, and it breaks the batch-folder mapping above.

The code expects different layouts in different places:

- **Batch folders:** `resolved_manifest.csv`, `src/baseline/path_resolver.py` and the Baseline adapter in `src/hueview/evaluate.py` (`BASELINE_FACE_DIRS`).
- **Either layout:** `src/hueview/landmark_extraction.py` walks the whole tree. `run_phase7_4_batch.py` and `run_phase7_5_batch.py` try the direct path first and fall back to scanning the batch folders.

Check which layout your copy of the data has before running a phase.

### Verified clean

`src/baseline/check_dataset.py` confirms **no cross-split leakage** — no image appears in more than one of train/val/test. The Phase 5 splits are sound and must not be regenerated; the methodology requires both models reuse them unchanged.

---

## Notes for whoever picks this up

**Do not normalize images before the CNN.** Keras' EfficientNet expects pixels in `[0, 255]` and rescales inside the model graph. Dividing by 255 first normalizes twice and quietly costs accuracy. The 6.1 RGB features *are* scaled to `[0, 1]`, but those feed the other branch — the two are different inputs with different conventions.

**The RGB branch is outnumbered 1280-to-3.** Straight concatenation is what the methodology specifies, so that's the default, but the CNN branch dominates the gradient almost entirely. `src/baseline/model.py` exposes `rgb_projection_dim` if this is ever revisited — with the caveat that HueView's fusion needs the same treatment, or the comparison stops being like-for-like.

**Check the undertone distribution before it reaches the results chapter.** The Neutral band spans only 0.01 (b_ratio 0.275–0.285) and skin tones cluster tightly. If one class dominates, that is a finding to report plainly, not three categories to present as equally exercised. `src/baseline/undertone.py` prints the distribution and flags any class above 90%.

**SCC label order is shared across three places.** `frontend/src/constants.js`, `ml_pipeline/configs/scc_labels.json` and the trained models' output order must agree. A mismatch mislabels every prediction without erroring.

**Trained weights are gitignored** (`ml_pipeline/models/`, `*.h5`, `*.keras`, `*.pkl`). They live on Google Drive. Agree on a location before someone's laptop dies.
