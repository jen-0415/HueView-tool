// Mock payload. Same shape as the real /api/analyze response,
// so switching VITE_USE_MOCK to false changes nothing in the UI.

export const MOCK_DETECT = {
  confidence: 0.97,
  landmarks_found: 5,
  bbox: [120, 84, 310, 310],
  upscaled: false,
  crop_preview: null, // the real API returns a base64 image here
};

export const MOCK_RESULT = {
  image: { width: 224, height: 224, preview: null },
  detection: { confidence: 0.97, landmarks_found: 5, upscaled: false },

  models: {
    baseline: {
      name: "Baseline",
      method: "Global RGB averaging",
      checkpoint: "b6f1a2c",
      placeholder: false,
      scc: "SCC-4",
      probabilities: [0.03, 0.08, 0.14, 0.71, 0.03, 0.01],
      confidence: 0.71,
      margin: 0.57,
      // Baseline works in RGB. No CIELAB block here — the conversion belongs
      // to HueView, and carrying a lab field would invite it back onto the screen.
      rgb: { R: 168, G: 132, B: 118 },
      regions: null,
      // Baseline rule: normalized RGB ratios, thresholded on the b ratio.
      //   b > 0.285          -> Cool
      //   0.275 <= b <= 0.285 -> Neutral
      //   b < 0.275          -> Warm
      undertone: {
        method: "rgb_ratio",
        label: "Neutral",
        ratios: { r: 0.4019, g: 0.3158, b: 0.2823 },
        b_ratio: 0.2823,
        source: "Global RGB mean",
        distribution: null,
      },
      inference_ms: 41,
    },
    hueview: {
      name: "HueView",
      method: "SSR + regional segmentation",
      checkpoint: "9de4471",
      placeholder: true, // true until the Phase 10 checkpoint exists
      scc: "SCC-3",
      probabilities: [0.01, 0.05, 0.87, 0.05, 0.01, 0.01],
      confidence: 0.87,
      margin: 0.82,
      lab: { L: 62.4, a: 14.2, b: 18.7 },
      regions: [
        { name: "Forehead", L: 64.1, a: 12.8, b: 17.2, pixels: 8421, used: true },
        { name: "Left cheek", L: 61.9, a: 15.1, b: 19.4, pixels: 6210, used: true },
        { name: "Right cheek", L: 62.7, a: 14.6, b: 18.9, pixels: 6044, used: true },
        { name: "Nose bridge", L: 63.8, a: 13.9, b: 18.1, pixels: 3102, used: true },
        { name: "Jawline", L: 59.4, a: 14.9, b: 17.8, pixels: 4870, used: true },
        { name: "Full face", L: 62.4, a: 14.2, b: 18.7, pixels: 28647, used: false },
      ],
      // HueView rule: hue angle on the regional CIELAB means.
      undertone: {
        method: "hue_angle",
        label: "Warm",
        hue_angle_deg: 52.8,
        source: "Mean of the selected regional CIELAB values",
        // Share of selected regions falling in each undertone band.
        distribution: { warm: 0.74, neutral: 0.19, cool: 0.07 },
      },
      inference_ms: 386,
    },
  },

  comparison: { agreement: false, scc_delta: 1, confidence_delta: 0.16 },

  test_set_metrics: {
    available: false,
    source: "Phase 11 held-out test split",
  },
};