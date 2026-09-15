// All screen transitions live here so the screens stay presentational.

import { useState, useCallback } from "react";
import { detect, analyze } from "./api";

export function useAnalysis() {
  const [screen, setScreen] = useState("upload"); // upload | confirm | analyzing | results
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [detection, setDetection] = useState(null);
  const [stages, setStages] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const upload = useCallback(async (f) => {
    setError(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setBusy(true);
    try {
      setDetection(await detect(f));
      setScreen("confirm");
    } catch (e) {
      setError(e.message);
      setPreview(null);
    } finally {
      setBusy(false);
    }
  }, []);

  const run = useCallback(() => {
    setStages({});
    setError(null);
    setScreen("analyzing");
    analyze(
      file,
      (s) => setStages((prev) => ({ ...prev, [s.key]: s })),
      (r) => {
        setResult(r);
        setScreen("results");
      },
      (e) => {
        setError(e.message);
        setScreen("confirm");
      }
    );
  }, [file]);

  const reset = useCallback(() => {
    setFile(null);
    setPreview(null);
    setDetection(null);
    setStages({});
    setResult(null);
    setError(null);
    setScreen("upload");
  }, []);

  return { screen, preview, detection, stages, result, error, busy, upload, run, reset };
}
