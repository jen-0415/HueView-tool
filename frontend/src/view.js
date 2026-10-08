// Which page is open.
//
// The tab is kept in the URL hash ("#analyze", "#evaluation") so the
// evaluation results can be linked to directly. `right: true` puts a tab at
// the far end of the header bar instead of next to the others.

import { useCallback, useEffect, useState } from "react";

export const TABS = [
  { id: "analyze", label: "Analyze a photo" },
  { id: "evaluation", label: "Evaluation Results", right: true },
];

const ids = TABS.map((t) => t.id);

function readHash() {
  // Tolerates the older "#<tab>/<something>" form.
  const [tab] = window.location.hash.replace(/^#/, "").split("/");
  return ids.includes(tab) ? tab : "analyze";
}

export function useView() {
  const [tab, setTabState] = useState(readHash);

  // Keep the URL in step without touching history.state -- useAnalysis keeps
  // the current screen in there for the Back button.
  useEffect(() => {
    window.history.replaceState(window.history.state, "", `#${tab}`);
  }, [tab]);

  const setTab = useCallback((t) => setTabState(ids.includes(t) ? t : "analyze"), []);

  return { tab, setTab };
}
