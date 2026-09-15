// The only file that talks to the backend.
// While VITE_USE_MOCK=true nothing here touches the network.

import { MOCK_DETECT, MOCK_RESULT } from "./mockData";
import { STAGES } from "./constants";

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---------------------------------------------------------------- */
/* 1. Detect a face and return the crop preview                      */
/* ---------------------------------------------------------------- */
export async function detect(file) {
  if (USE_MOCK) {
    await sleep(700);
    return MOCK_DETECT;
  }

  const body = new FormData();
  body.append("image", file);

  const res = await fetch(`${BASE}/api/detect`, { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail?.message || "Face detection failed.");
  }
  return res.json();
}

/* ---------------------------------------------------------------- */
/* 2. Run both pipelines, streaming stage progress                   */
/*    onStage({ key, status, ms })  -> fired six times               */
/*    onResult(payload)             -> fired once at the end         */
/* ---------------------------------------------------------------- */
export function analyze(file, onStage, onResult, onError) {
  if (USE_MOCK) return mockAnalyze(onStage, onResult);

  const body = new FormData();
  body.append("image", file);

  fetch(`${BASE}/api/analyze`, { method: "POST", body })
    .then((r) => r.json())
    .then(({ job_id }) => {
      const es = new EventSource(`${BASE}/api/analyze/${job_id}/events`);

      es.addEventListener("stage", (e) => onStage(JSON.parse(e.data)));
      es.addEventListener("result", (e) => {
        onResult(JSON.parse(e.data).data);
        es.close();
      });
      es.addEventListener("error", (e) => {
        onError(new Error("Analysis failed. Check that the API is running."));
        es.close();
      });
    })
    .catch(() => onError(new Error("Could not reach the API.")));
}

/* Fake stream so the UI is fully demo-able with no backend. */
async function mockAnalyze(onStage, onResult) {
  const timings = [420, 560, 900, 1100, 380, 440];
  for (let i = 0; i < STAGES.length; i++) {
    onStage({ key: STAGES[i].key, status: "running" });
    await sleep(timings[i]);
    onStage({ key: STAGES[i].key, status: "done", ms: timings[i] });
  }
  await sleep(300);
  onResult(MOCK_RESULT);
}
