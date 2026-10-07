# HueView frontend

React + Vite + Tailwind. Uploads a face photo to the HueView API and shows the Baseline-vs-HueView comparison: face check, SSR intermediate images, both models' SCC and undertone, colour values, and HueView's regional segmentation with per-region SCC.

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
| `src/api.js` | the only file that talks to the backend (`/api/detect`, `/api/analyze` + SSE events) |
| `src/useAnalysis.js` | screen flow: upload → confirm → analyzing (SSR preview, Continue) → results |
| `src/constants.js` | SCC labels and colours, undertone rule text (must match the backend order) |
| `src/mockData.js` | illustrative mock response, used only in mock mode |
| `src/screens/` | `Upload`, `Confirm`, `Analyzing`, `Results` |
| `src/components/` | `ModelCard`, `ColorValuesCard`, `UndertoneCard`, `IlluminationCard`, `RegionalSegmentationCard`, `RegionTable`, … |

All displayed predictions and values come from the API response; nothing is computed in the browser except the agreement banner.
