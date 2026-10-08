// The six SCC classes. Order must match the backend (configs/scc_labels.json
// and the models' output order).
//
// SCC stands for Skin Color Clusters: the six groups this study classifies
// into. They are not a scale of their own -- each one is a group of steps on
// the 10-tone Monk Skin Tone (MST) scale, which is how every photo in the
// dataset was labelled. The `mst` field below is that grouping, and it must
// stay identical to MST_TO_SCC in
// ml_pipeline/src/preprocessing/add_scc_labels.py, which produced the ground
// truth the whole study is scored against.
export const SCC = [
  { id: "SCC-1", label: "Very Light", hex: "#F6E3D2", mst: "MST 1–2" },
  { id: "SCC-2", label: "Light", hex: "#E0BE99", mst: "MST 3–4" },
  { id: "SCC-3", label: "Medium", hex: "#B87A50", mst: "MST 5–6" },
  { id: "SCC-4", label: "Olive", hex: "#8B5C3C", mst: "MST 7" },
  { id: "SCC-5", label: "Brown", hex: "#4E2D1D", mst: "MST 8–9" },
  { id: "SCC-6", label: "Deep", hex: "#2A1510", mst: "MST 10" },
];

export const sccLabel = (id) => SCC.find((s) => s.id === id)?.label ?? id;
export const sccMst = (id) => SCC.find((s) => s.id === id)?.mst ?? null;

// What SCC means, in one place so the upload screen and the result screen
// cannot explain the scale differently.
export const SCC_MEANING = {
  name: "Skin Color Clusters",
  short:
    "SCC stands for Skin Color Clusters — the six skin tone groups this study classifies into.",
  basis:
    "The groups come from the Monk Skin Tone (MST) scale, a 10-step scale from lightest to " +
    "deepest. Every photo in the dataset carries an MST label from 1 to 10, and neighbouring " +
    "steps were merged into these six clusters so that each one has enough photos to train on.",
};

// Undertone rules, one per pipeline. Keep these in sync with the manuscript —
// the backend is the source of truth, this is for display only.
export const UNDERTONE_RULES = {
  rgb_ratio: {
    title: "Normalized RGB ratios",
    formula: "r = R/(R+G+B),  g = G/(R+G+B),  b = B/(R+G+B)",
    bands: [
      { label: "Cool", test: "b > 0.285" },
      { label: "Neutral", test: "0.275 \u2264 b \u2264 0.285" },
      { label: "Warm", test: "b < 0.275" },
    ],
  },
  hue_angle: {
    title: "Hue angle on CIELAB",
    formula: "h\u2090\u1d66 = atan2(b*, a*)",
    bands: [],
  },
};