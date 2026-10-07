// All screen transitions live here so the screens stay presentational.
//
// Screen changes now also push browser history entries, so the Back button
// steps back through the app instead of leaving it. "analyzing" is treated
// as transient: moving to "results" REPLACES it in history, so Back from
// results lands on confirm rather than a stale progress screen.

import { useState, useCallback, useEffect, useRef } from "react";
import { detect, analyze } from "./api";

export function useAnalysis() {
  const [screen, setScreen] = useState("upload"); // upload | confirm | analyzing | results
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [detection, setDetection] = useState(null);
  const [stages, setStages] = useState({});
  const [result, setResult] = useState(null);
  const [ssrPreview, setSsrPreview] = useState(null); // { original, ssr } data URLs
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  // When an SSR preview was shown, the result waits here until the user
  // clicks Continue, so the normalization step is always seen.
  const [pendingResult, setPendingResult] = useState(null);
  const sawSsr = useRef(false);

  // Holds the cancel function for an in-flight analysis so leaving the
  // analyzing screen can close the SSE connection instead of letting its
  // callbacks fire after the user has navigated away.
  const cancelAnalysis = useRef(null);

  const clearAll = useCallback(() => {
    setFile(null);
    setPreview(null);
    setDetection(null);
    setStages({});
    setResult(null);
    setSsrPreview(null);
    setPendingResult(null);
    setError(null);
  }, []);

  // Navigate forward, recording a history entry.
  const goTo = useCallback((next, { replace = false } = {}) => {
    setScreen(next);
    const entry = { hueviewScreen: next };
    if (replace) window.history.replaceState(entry, "");
    else window.history.pushState(entry, "");
  }, []);

  // Seed the initial history entry, and handle Back/Forward.
  useEffect(() => {
    window.history.replaceState({ hueviewScreen: "upload" }, "");

    const onPopState = (e) => {
      const target = e.state?.hueviewScreen ?? "upload";

      // Leaving analyzing (or anywhere) means any running analysis should stop.
      cancelAnalysis.current?.();
      cancelAnalysis.current = null;

      setScreen(target);

      if (target === "upload") clearAll();
      if (target === "confirm") {
        setStages({});
        setResult(null);
        setSsrPreview(null);
        setPendingResult(null);
        setError(null);
      }
    };

    window.addEventListener("popstate", onPopState);
    return () => {
      window.removeEventListener("popstate", onPopState);
      cancelAnalysis.current?.();
    };
  }, [clearAll]);

  const upload = useCallback(
    async (f) => {
      setError(null);
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setBusy(true);
      try {
        setDetection(await detect(f));
        goTo("confirm");
      } catch (e) {
        setError(e.message);
        setPreview(null);
      } finally {
        setBusy(false);
      }
    },
    [goTo]
  );

  const run = useCallback(() => {
    setStages({});
    setError(null);
    setSsrPreview(null);
    setPendingResult(null);
    sawSsr.current = false;
    goTo("analyzing");

    cancelAnalysis.current = analyze(
      file,
      (s) => setStages((prev) => ({ ...prev, [s.key]: s })),
      (r) => {
        cancelAnalysis.current = null;
        if (sawSsr.current) {
          setPendingResult(r); // shown when the user clicks Continue
          return;
        }
        setResult(r);
        // replace: analyzing is transient, so Back from results goes to confirm
        goTo("results", { replace: true });
      },
      (e) => {
        cancelAnalysis.current = null;
        setError(e.message);
        goTo("confirm", { replace: true });
      },
      (p) => {
        sawSsr.current = true;
        setSsrPreview(p);
      }
    );
  }, [file, goTo]);

  const showResults = useCallback(() => {
    if (!pendingResult) return;
    setResult(pendingResult);
    setPendingResult(null);
    goTo("results", { replace: true });
  }, [pendingResult, goTo]);

  const reset = useCallback(() => {
    cancelAnalysis.current?.();
    cancelAnalysis.current = null;
    clearAll();
    goTo("upload");
  }, [clearAll, goTo]);

  return {
    screen, preview, detection, stages, result, ssrPreview,
    resultReady: pendingResult !== null,
    error, busy, upload, run, showResults, reset,
  };
}