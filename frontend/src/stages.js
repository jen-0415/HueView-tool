// Result stages, in the order the pipeline runs them. Shared by the sidebar
// and the Results screen's prev/next buttons.
export const STEPS = [
  { id: "detect", n: "01", label: "Face detection", scope: "shared by both pipelines" },
  { id: "illum", n: "02", label: "Illumination bin", scope: "shared by both pipelines" },
  { id: "ssr", n: "03", label: "Lighting correction", scope: "HueView only" },
  { id: "seg", n: "04", label: "Face regions", scope: "HueView only" },
  { id: "skin", n: "05", label: "Skin filter", scope: "HueView only" },
  { id: "region", n: "06", label: "Per-region SCC", scope: "HueView only" },
  { id: "decision", n: "07", label: "Final decision", scope: "HueView only" },
  { id: "under", n: "08", label: "Undertone", scope: "exploratory" },
];

export const ORDER = [
  { id: "summary", label: "Summary" },
  ...STEPS,
  { id: "compare", label: "Baseline vs HueView" },
];
