import Card from "../components/Card";
import ModelCard from "../components/ModelCard";
import ColorValuesCard from "../components/ColorValuesCard";
import IlluminationCard from "../components/IlluminationCard";
import UndertoneCard from "../components/UndertoneCard";
import RegionTable from "../components/RegionTable";
import RegionalSegmentationCard from "../components/RegionalSegmentationCard";
import { sccLabel } from "../constants";

export default function Results({ preview, result, onReset }) {
  const { baseline, hueview } = result.models;
  // A null SCC means that model's weights aren't loaded -- never call two
  // missing predictions an agreement.
  const bothPredicted = Boolean(baseline.scc && hueview.scc);
  const agree = bothPredicted && baseline.scc === hueview.scc;
  const sccText = (m) => (m.scc ? `${m.scc} — ${sccLabel(m.scc)}` : "No prediction");

  return (
    <main className="max-w-6xl mx-auto px-8 py-10">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="font-display italic text-3xl">Analysis results</h2>
          <p className="font-mono text-[11px] text-ink-soft mt-2">
            EfficientNetB0 · STW · baseline {baseline.checkpoint} · hueview{" "}
            {hueview.checkpoint}
          </p>
        </div>
        <button
          onClick={onReset}
          className="rounded-full bg-white border border-line text-accent text-sm px-5 py-2"
        >
          + New analysis
        </button>
      </div>

      {/* The headline finding. Do not bury this in a table. */}
      {bothPredicted ? (
        <div
          className={`rounded-2xl px-6 py-4 mt-6 flex flex-wrap items-center gap-x-4 gap-y-1 border ${agree ? "bg-ok-bg border-ok" : "bg-blush-soft border-accent"
            }`}
        >
          <span
            className={`font-display text-xl ${agree ? "text-ok" : "text-accent"}`}
          >
            {agree ? "Both pipelines agree" : "The pipelines disagree"}
          </span>
          <span className="text-sm text-ink-soft">
            {agree
              ? `Both classified this face as ${baseline.scc} — ${sccLabel(baseline.scc)}.`
              : `Baseline read this face as ${baseline.scc} — ${sccLabel(
                baseline.scc,
              )}; HueView read it as ${hueview.scc} — ${sccLabel(hueview.scc)}.`}
          </span>
        </div>
      ) : (
        <div className="rounded-2xl px-6 py-4 mt-6 flex flex-wrap items-center gap-x-4 gap-y-1 border bg-warm border-warm-ink/30">
          <span className="font-display text-xl text-warm-ink">Comparison unavailable</span>
          <span className="text-sm text-ink-soft">
            {[baseline, hueview]
              .filter((m) => !m.scc)
              .map((m) => m.name)
              .join(" and ")}{" "}
            returned no SCC prediction (model weights not loaded), so the two pipelines
            can&apos;t be compared for this image.
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[220px_1fr_1fr] gap-6 mt-6 items-start">
        <div>
          <img
            src={result.image.preview || preview}
            alt=""
            className="w-[220px] h-[220px] object-cover rounded-lg"
          />
          <p className="font-mono text-[10px] text-ink-soft mt-3">
            224 × 224 crop · {result.detection.landmarks_found}/5 landmarks
          </p>
        </div>

        <ModelCard model={baseline} />
        <ModelCard model={hueview} primary />
      </div>

      <div className="mt-6">
        <RegionalSegmentationCard hueview={hueview} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6 mt-6 items-start">
        <IlluminationCard illumination={result.illumination} />

        <Card>
          <div className="px-6 py-4 border-b border-line-soft">
            <div className="font-semibold">Per-image comparison</div>
            <div className="text-xs text-ink-soft">
              Single-image outputs from the two pipelines on the same 224 × 224
              crop.
            </div>
          </div>

          <table className="w-full text-sm">
            <thead>
              <tr className="bg-blush">
                {["Measure", "Baseline", "HueView"].map((h) => (
                  <th
                    key={h}
                    className="px-5 py-3 text-left font-mono text-[10px] tracking-widest text-accent font-normal"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <Row
                k="Input representation"
                a="EfficientNetB0 on the full crop + global RGB mean"
                b="Five skin-masked SSR regional patches + SSR full face, each with CIELAB"
              />
              <Row
                k="Predicted class"
                a={sccText(baseline)}
                b={sccText(hueview)}
                highlight
              />
              <Row k="Color space" a="RGB" b="CIELAB" />
              <Row
                k="Undertone rule"
                a="Normalized RGB ratios, b-ratio threshold"
                b="Hue angle on regional CIELAB means"
              />
              <Row
                k="Undertone descriptor"
                a={`${baseline.undertone.label} (b = ${baseline.undertone.b_ratio})`}
                b={`${hueview.undertone.label ?? "—"} (mean regional hue ${hueview.undertone.hue_angle_deg ?? "—"}°)`}
              />
            </tbody>
          </table>
        </Card>
      </div>

      <div className="mt-6">
        <ColorValuesCard baseline={baseline} hueview={hueview} />
      </div>

      <div className="mt-6">
        <UndertoneCard baseline={baseline} hueview={hueview} />
      </div>

      <div className="mt-6">
        <RegionTable regions={hueview.regions} />
      </div>
    </main>
  );
}

function Row({ k, a, b, highlight }) {
  return (
    <tr className="border-t border-line-soft">
      <td className="px-5 py-3 text-ink-soft">{k}</td>
      <td className={`px-5 py-3 ${highlight ? "font-semibold" : ""}`}>{a}</td>
      <td
        className={`px-5 py-3 ${highlight ? "font-semibold text-accent" : ""}`}
      >
        {b}
      </td>
    </tr>
  );
}