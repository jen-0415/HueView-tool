// The six SCC classes. Order must match the backend (configs/scc_labels.json
// and the models' output order).
export const SCC = [
  { id: "SCC-1", label: "Very Light", hex: "#F6E3D2" },
  { id: "SCC-2", label: "Light", hex: "#E0BE99" },
  { id: "SCC-3", label: "Medium", hex: "#B87A50" },
  { id: "SCC-4", label: "Olive", hex: "#8B5C3C" },
  { id: "SCC-5", label: "Brown", hex: "#4E2D1D" },
  { id: "SCC-6", label: "Deep", hex: "#2A1510" },
];

export const sccLabel = (id) => SCC.find((s) => s.id === id)?.label ?? id;

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