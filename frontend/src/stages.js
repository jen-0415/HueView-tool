// Result stages, in the order the pipeline runs them. Shared by the sidebar
// and the Results screen's prev/next buttons.
export const STEPS = [
  { id: "detect", n: "01", label: "Face detection", scope: "shared by both pipelines" },
  { id: "illum", n: "02", label: "Lighting level", scope: "shared by both pipelines" },
  { id: "ssr", n: "03", label: "Lighting correction", scope: "ChroLumi only" },
  { id: "seg", n: "04", label: "Face regions", scope: "ChroLumi only" },
  { id: "skin", n: "05", label: "Skin pixels", scope: "ChroLumi only" },
  { id: "region", n: "06", label: "Region results", scope: "ChroLumi only" },
  { id: "decision", n: "07", label: "Final decision", scope: "ChroLumi only" },
  { id: "under", n: "08", label: "Undertone", scope: "experimental" },
];

export const ORDER = [
  { id: "summary", label: "Summary" },
  ...STEPS,
  { id: "compare", label: "Baseline vs ChroLumi" },
];
