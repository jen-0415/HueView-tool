// GET /api/evaluation, shared by the two evaluation views so both read the
// same saved result files and can never disagree on a number.

import { useEffect, useState } from "react";
import { getEvaluation } from "./api";

export function useEvaluation() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let live = true;
    getEvaluation().then(
      (d) => live && setData(d),
      (e) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, []);

  return { data, error };
}
