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
/*                                                                   */
/*    Returns a cancel() function. Call it when leaving the          */
/*    analyzing screen (browser Back, reset) so the SSE connection   */
/*    closes and no callbacks fire after the user has moved on.      */
/* ---------------------------------------------------------------- */
export function analyze(file, onStage, onResult, onError) {
  if (USE_MOCK) return mockAnalyze(onStage, onResult);

  let cancelled = false;
  let es = null;

  const body = new FormData();
  body.append("image", file);

  fetch(`${BASE}/api/analyze`, { method: "POST", body })
    .then((r) => r.json())
    .then(({ job_id }) => {
      if (cancelled) return; // user left before the job id came back

      es = new EventSource(`${BASE}/api/analyze/${job_id}/events`);

      es.addEventListener("stage", (e) => {
        if (!cancelled) onStage(JSON.parse(e.data));
      });
      es.addEventListener("result", (e) => {
        if (!cancelled) onResult(JSON.parse(e.data).data);
        es.close();
      });
      es.addEventListener("error", () => {
        if (!cancelled) {
          onError(new Error("Analysis failed. Check that the API is running."));
        }
        es.close();
      });
    })
    .catch(() => {
      if (!cancelled) onError(new Error("Could not reach the API."));
    });

  return () => {
    cancelled = true;
    es?.close();
  };
}

/* Fake stream so the UI is fully demo-able with no backend. */
function mockAnalyze(onStage, onResult) {
  let cancelled = false;

  (async () => {
    const timings = [420, 560, 900, 1100, 380, 440];
    for (let i = 0; i < STAGES.length; i++) {
      if (cancelled) return;
      onStage({ key: STAGES[i].key, status: "running" });
      await sleep(timings[i]);
      if (cancelled) return;
      onStage({ key: STAGES[i].key, status: "done", ms: timings[i] });
    }
    await sleep(300);
    if (!cancelled) onResult(MOCK_RESULT);
  })();

  return () => {
    cancelled = true;
  };
}