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

// The stage events the API streams while an analysis runs, in the order
// routes.py sends them. Keys must match STAGE_KEYS in
// ml_pipeline/src/api/inference.py, or the progress list silently stops
// advancing. These are NOT the result stages above: the result stages are what
// the user can browse afterwards, these are what is happening right now.
export const ANALYSIS_STAGES = [
  { key: "detect", label: "Finding the face" },
  { key: "ssr", label: "Correcting the lighting" },
  { key: "segment", label: "Outlining the face areas" },
  { key: "cnn", label: "Classifying each area" },
  { key: "lab", label: "Reading the colour values" },
  { key: "undertone", label: "Estimating the undertone" },
];
