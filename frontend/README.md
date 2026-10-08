# HueView frontend

React + Vite + Tailwind. Uploads a face photo to the HueView API and shows the Baseline-vs-HueView comparison: face check, SSR intermediate images, both models' SCC and undertone, colour values, and HueView's regional segmentation with per-region SCC. It also serves the evaluation results read from the saved result files.

## Tabs

Two, listed in `src/view.js`: **Analyze a photo** (left of the header) and **Evaluation Results** (far right). The open tab is kept in the URL hash, so `#evaluation` links straight to the results. Switching tabs never re-runs an analysis — `useAnalysis` sits above the tab state in `App.jsx`.

`screens/Evaluation.jsx` shows the Baseline and HueView under each lighting level and HueView per facial region, with the confusion matrices and per-class results. It lays out only what `evaluate.py` and `appendix_tables.py` wrote; nothing is recomputed in the browser. The significance tests the API still returns under `significance` are not displayed.

## SCC

`src/constants.js` holds the six classes, their display colours, and the MST steps each one covers; `components/SccScaleCard.jsx` and the upload screen's reference scale both explain the scale from those constants. The `mst` field must stay identical to `MST_TO_SCC` in `ml_pipeline/src/preprocessing/add_scc_labels.py`, which is what produced the ground truth:

| SCC | Label | MST steps |
|---|---|---|
| SCC-1 | Very Light | 1–2 |
| SCC-2 | Light | 3–4 |
| SCC-3 | Medium | 5–6 |
| SCC-4 | Olive | 7 |
| SCC-5 | Brown | 8–9 |
| SCC-6 | Deep | 10 |

## Run

```powershell
npm install
copy .env.example .env   # optional
npm run dev              # http://localhost:5173
```

Start the API first (`cd ../ml_pipeline; uvicorn src.api.main:app --port 8000 --reload`). The API's CORS allows only port 5173, so `npm run preview` (port 4173) cannot reach it as configured.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `VITE_USE_MOCK` | unset (= false) | `true` serves `src/mockData.js` and never calls the API |
| `VITE_API_BASE` | `http://localhost:8000` | API base URL |

## Layout

| Path | Role |
|---|---|
| `src/api.js` | the only file that talks to the backend (`/api/detect`, `/api/analyze` + SSE events, `/api/evaluation`) |
| `src/useAnalysis.js` | screen flow: upload → confirm → analyzing (SSR preview, Continue) → results |
| `src/useEvaluation.js` | `GET /api/evaluation`, called only by the Evaluation screen |
| `src/view.js` | the tabs and the URL-hash plumbing |
| `src/constants.js` | SCC labels, colours and MST ranges, undertone rule text (must match the backend order) |
| `src/mockData.js` | illustrative mock response, used only in mock mode |
| `src/screens/` | `Upload`, `Confirm`, `Analyzing`, `Results`, `Evaluation` |
| `src/components/` | `ModelCard`, `ColorValuesCard`, `UndertoneCard`, `IlluminationCard`, `RegionalSegmentationCard`, `RegionTable`, `SccScaleCard`, … |

All displayed predictions and values come from the API response; nothing is computed in the browser except the agreement banner.
