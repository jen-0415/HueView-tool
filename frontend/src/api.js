// The only file that talks to the backend.
//
// While VITE_USE_MOCK=true, nothing here touches the network.

import { MOCK_DETECT, MOCK_RESULT } from "./mockData";

const BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

export const USE_MOCK =
  import.meta.env.VITE_USE_MOCK === "true";

const sleep = (ms) =>
  new Promise((r) => setTimeout(r, ms));

/* ----------------------------------------------------------------
   1. Detect a face and return the crop preview
   ---------------------------------------------------------------- */

export async function detect(file) {
  // Mock mode
  if (USE_MOCK) {
    await sleep(700);
    return MOCK_DETECT;
  }

  // Real backend
  const body = new FormData();
  body.append("image", file);

  const res = await fetch(`${BASE}/api/detect`, {
    method: "POST",
    body,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));

    throw new Error(
      err?.detail?.message || "Face detection failed."
    );
  }

  return res.json();
}

/* ----------------------------------------------------------------
   2. Run both pipelines, streaming stage progress

   onStage({ key, status, ms })
   onSsr({ original, ssr, steps })   -- PNG data URLs, sent before the result;
                                        steps = [{ label, image }] or null
   onResult(payload)

   Mock mode sends no SSR preview.

   Returns a cancel() function.
   ---------------------------------------------------------------- */

export function analyze(
  file,
  onStage,
  onResult,
  onError,
  onSsr = () => {}
) {
  // Mock mode
  if (USE_MOCK) {
    return mockAnalyze(onStage, onResult);
  }

  // Real backend
  let cancelled = false;
  let es = null;

  const body = new FormData();
  body.append("image", file);

  fetch(`${BASE}/api/analyze`, {
    method: "POST",
    body,
  })
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));

        throw new Error(
          err?.detail?.message || "Analysis request failed."
        );
      }

      return res.json();
    })
    .then(({ job_id }) => {
      if (cancelled) return;

      // Open Server-Sent Events connection
      es = new EventSource(
        `${BASE}/api/analyze/${job_id}/events`
      );

      // Analysis stage updates
      es.addEventListener("stage", (e) => {
        if (cancelled) return;

        try {
          onStage(JSON.parse(e.data));
        } catch {
          console.error("Invalid stage event:", e.data);
        }
      });

      // SSR preview: the cropped face before and after SSR
      es.addEventListener("ssr", (e) => {
        if (cancelled) return;

        try {
          onSsr(JSON.parse(e.data));
        } catch {
          console.error("Invalid ssr event:", e.data);
        }
      });

      // Final analysis result
      es.addEventListener("result", (e) => {
        if (cancelled) return;

        try {
          const payload = JSON.parse(e.data);
          onResult(payload.data);
        } catch {
          onError(
            new Error("Invalid analysis result received.")
          );
        }

        es.close();
        es = null;
      });

      // SSE connection error
      es.addEventListener("error", () => {
        if (!cancelled) {
          onError(
            new Error(
              "Analysis failed. Check that the API is running."
            )
          );
        }

        es?.close();
        es = null;
      });
    })
    .catch((err) => {
      if (!cancelled) {
        onError(
          new Error(
            err.message || "Could not reach the API."
          )
        );
      }
    });

  // Cancel function
  return () => {
    cancelled = true;
    es?.close();
    es = null;
  };
}

/* ----------------------------------------------------------------
   Mock analysis
   ---------------------------------------------------------------- */

function mockAnalyze(onStage, onResult) {
  let cancelled = false;

  const stages = [
    "detect",
    "crop",
    "normalize",
    "extract",
    "predict",
    "compare",
  ];

  const timings = [
    420,
    560,
    900,
    1100,
    380,
    440,
  ];

  (async () => {
    for (let i = 0; i < stages.length; i++) {
      if (cancelled) return;

      onStage({
        key: stages[i],
        status: "running",
      });

      await sleep(timings[i]);

      if (cancelled) return;

      onStage({
        key: stages[i],
        status: "done",
        ms: timings[i],
      });
    }

    await sleep(300);

    if (!cancelled) {
      onResult(MOCK_RESULT);
    }
  })();

  return () => {
    cancelled = true;
  };
}