import { useState } from "react";
import Card from "../components/Card";
import ModelCard from "../components/ModelCard";
import ColorValuesCard from "../components/ColorValuesCard";
import IlluminationCard from "../components/IlluminationCard";
import UndertoneCard from "../components/UndertoneCard";
import RegionTable from "../components/RegionTable";
import ResultsSidebar from "../components/ResultsSidebar";
import {
  SegmentationFigures,
  RegionPredictions,
  DecisionTable,
} from "../components/RegionalSegmentationCard";
import { sccLabel } from "../constants";
import { STEPS, ORDER } from "../stages";

// One stage at a time, picked from the sidebar. Each stage panel shows what
// that step did ("What happened") and a plain-language reading of this
// photo's values ("What this means"), written from the real result.
export default function Results({ preview, result, ssrPreview, onReset }) {
  const [stage, setStage] = useState("summary");
  const idx = ORDER.findIndex((o) => o.id === stage);

  const pick = (id) => {
    setStage(id);
    window.scrollTo({ top: 0 });
  };

  const props = { result, ssrPreview, preview, pick };

  return (
    <main className="max-w-[1360px] mx-auto px-4 sm:px-8 py-7 flex flex-col lg:flex-row gap-7 items-start">
      <ResultsSidebar
        stage={stage}
        onPick={pick}
        preview={result.image.preview || preview}
        onReset={onReset}
      />

      <section className="flex-1 min-w-0 w-full flex flex-col gap-5">
        {stage === "summary" && <SummaryPanel {...props} />}
        {stage === "detect" && <DetectPanel {...props} />}
        {stage === "illum" && <IlluminationPanel {...props} />}
        {stage === "ssr" && <SsrPanel {...props} />}
        {stage === "seg" && <SegmentationPanel {...props} />}
        {stage === "skin" && <SkinPanel {...props} />}
        {stage === "region" && <RegionPanel {...props} />}
        {stage === "decision" && <DecisionPanel {...props} />}
        {stage === "under" && <UndertonePanel {...props} />}
        {stage === "compare" && <ComparePanel {...props} />}

        <div className="flex justify-between gap-3 flex-wrap border-t border-line pt-4">
          <div>
            {idx > 0 && (
              <button
                onClick={() => pick(ORDER[idx - 1].id)}
                className="rounded-full bg-white border border-line text-accent text-sm px-4 min-h-[44px]"
              >
                ← {ORDER[idx - 1].label}
              </button>
            )}
          </div>
          <div>
            {idx < ORDER.length - 1 && (
              <button
                onClick={() => pick(ORDER[idx + 1].id)}
                className="rounded-full bg-accent text-white text-sm font-semibold px-5 min-h-[44px]"
              >
                {ORDER[idx + 1].label} →
              </button>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

/* ---------- shared bits ---------- */

const pct = (p, digits = 0) => `${(p * 100).toFixed(digits)}%`;
const sccText = (m) => (m.scc ? `${m.scc} — ${sccLabel(m.scc)}` : "No prediction");

function StageHeader({ id, title }) {
  const step = STEPS.find((s) => s.id === id);
  const eyebrow = step
    ? `Stage ${step.n} of ${String(STEPS.length).padStart(2, "0")} · ${step.scope}`
    : title;
  return (
    <div>
      <div className="font-mono text-[11px] tracking-widest uppercase text-ink-soft">{eyebrow}</div>
      <h2 className="font-display text-[32px] font-medium mt-1.5">{title}</h2>
    </div>
  );
}

function Happened({ children }) {
  return (
    <>
      <div className="font-mono text-[11px] tracking-widest uppercase text-ink-soft mb-1.5">
        What happened
      </div>
      <p className="text-sm leading-relaxed text-[#3A2A2E] mb-4">{children}</p>
    </>
  );
}

function Meaning({ children }) {
  return (
    <div className="bg-blush-soft border border-line rounded-[14px] px-5 py-4">
      <div className="font-mono text-[11px] tracking-widest uppercase text-accent">
        What this means
      </div>
      <div className="mt-2 text-[15px] leading-relaxed space-y-2">{children}</div>
    </div>
  );
}

function Unavailable({ what }) {
  return (
    <Card className="px-6 py-5 text-sm text-ink-soft">
      {what} isn&apos;t available for this result.
    </Card>
  );
}

function Chip({ onClick, children }) {
  return (
    <button
      onClick={onClick}
      className="rounded-full bg-white border border-line text-accent text-[13px] px-3.5 min-h-[36px]"
    >
      {children}
    </button>
  );
}

/* ---------- panels ---------- */

function SummaryPanel({ result, pick }) {
  const { baseline, hueview } = result.models;
  // A null SCC means that model's weights aren't loaded -- never call two
  // missing predictions an agreement.
  const bothPredicted = Boolean(baseline.scc && hueview.scc);
  const agree = bothPredicted && baseline.scc === hueview.scc;

  return (
    <>
      <div>
        <div className="font-mono text-[11px] tracking-widest uppercase text-ink-soft">Summary</div>
        <h2 className="font-display italic text-[34px] font-medium mt-1.5">Analysis results</h2>
        <p className="font-mono text-[11px] text-ink-soft mt-1">
          EfficientNetB0 · STW · baseline {baseline.checkpoint} · hueview {hueview.checkpoint}
        </p>
      </div>

      {/* The headline finding. Do not bury this in a table. */}
      {bothPredicted ? (
        <div
          className={`rounded-2xl px-6 py-4 flex flex-wrap items-center gap-x-4 gap-y-1 border ${
            agree ? "bg-ok-bg border-ok" : "bg-blush-soft border-accent"
          }`}
        >
          <span className={`font-display text-xl ${agree ? "text-ok" : "text-accent"}`}>
            {agree ? "Both pipelines agree" : "The pipelines disagree"}
          </span>
          <span className="text-sm text-ink-soft">
            {agree
              ? `Both classified this face as ${sccText(baseline)}.`
              : `Baseline read this face as ${sccText(baseline)}; HueView read it as ${sccText(hueview)}.`}
          </span>
        </div>
      ) : (
        <div className="rounded-2xl px-6 py-4 flex flex-wrap items-center gap-x-4 gap-y-1 border bg-warm border-warm-ink/30">
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

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5 items-start">
        <ModelCard model={baseline} />
        <ModelCard model={hueview} primary />
      </div>

      <Meaning>
        <p>
          The Baseline averages the whole original photo, including hair, background and
          shadows. HueView corrects the lighting first, then reads only skin pixels from
          separate face regions. Different inputs can lead to different classes.
        </p>
        <p>
          The undertone is an <b>exploratory</b> reading and was not validated against
          ground-truth labels.
        </p>
        <div className="flex flex-wrap gap-2 pt-2 items-center">
          <span className="text-[13px] text-ink-soft">See where the result comes from:</span>
          <Chip onClick={() => pick("ssr")}>03 Lighting correction →</Chip>
          <Chip onClick={() => pick("region")}>06 Per-region SCC →</Chip>
          <Chip onClick={() => pick("decision")}>07 Final decision →</Chip>
        </div>
      </Meaning>
    </>
  );
}

function DetectPanel({ result, preview }) {
  const d = result.detection;
  return (
    <>
      <StageHeader id="detect" title="Face detection" />
      <Card className="p-6 flex flex-wrap gap-7">
        <img
          src={result.image.preview || preview}
          alt="Cropped face"
          className="w-[224px] h-[224px] object-cover rounded-xl"
        />
        <div className="flex-[1_1_300px] min-w-0">
          <Happened>
            MTCNN found the face and five key points (two eye centres, the nose tip and two
            mouth corners). The face was then cropped and resized to {result.image.width} ×{" "}
            {result.image.height} pixels, the input size both models expect.
          </Happened>
          <Value k="Detection confidence" v={pct(d.confidence)}>
            How sure MTCNN is that this is a face.
          </Value>
          <Value k="Key points found" v={`${d.landmarks_found} / 5`}>
            Used to align and crop the face.
          </Value>
          {d.upscaled && (
            <Value k="Upscaled" v="yes">
              The face was small in the photo and was enlarged before cropping.
            </Value>
          )}
        </div>
      </Card>
      <Meaning>
        <p>
          The face was found with {pct(d.confidence)} confidence and {d.landmarks_found} of 5 key
          points, so the later stages work on the cropped face shown here.
        </p>
      </Meaning>
    </>
  );
}

function Value({ k, v, children }) {
  return (
    <div className="grid grid-cols-[170px_minmax(0,1fr)] gap-x-4 gap-y-1 py-3 border-t border-line-soft">
      <span className="font-mono text-xs text-ink-soft">{k}</span>
      <span className="font-mono text-base">{v}</span>
      <span className="col-start-2 text-[13px] text-ink-soft leading-snug">{children}</span>
    </div>
  );
}

function IlluminationPanel({ result }) {
  const il = result.illumination;
  return (
    <>
      <StageHeader id="illum" title="Illumination bin" />
      <div className="grid grid-cols-1 md:grid-cols-[320px_1fr] gap-5 items-start">
        <IlluminationCard illumination={il} />
        <Card className="p-6">
          <Happened>
            The photo&apos;s {il.metric.toLowerCase()} was measured and compared with three
            lighting groups found by k-means on the whole dataset. The group is used to report
            accuracy by lighting condition. <b>It does not change the prediction.</b>
          </Happened>
        </Card>
      </div>
      <Meaning>
        <p>
          This photo falls in the <b>{il.bin}</b> lighting group ({il.metric.toLowerCase()}{" "}
          {il.value} on a {il.scale[0]}–{il.scale[1]} scale).
        </p>
      </Meaning>
    </>
  );
}

function SsrPanel({ ssrPreview }) {
  // The backend omits `steps` if its step preview stops matching the real SSR
  // output; fall back to just before/after.
  const steps = ssrPreview
    ? ssrPreview.steps ?? [
        { label: "Cropped face (original)", image: ssrPreview.original },
        { label: "After SSR", image: ssrPreview.ssr },
      ]
    : null;

  return (
    <>
      <StageHeader id="ssr" title="Lighting correction (Single-Scale Retinex)" />
      {steps ? (
        <Card className="p-6">
          <Happened>
            SSR estimates the lighting with a heavy blur of the photo, then divides it out. What
            remains is closer to the skin&apos;s own color, with fewer shadows and uneven
            highlights.
          </Happened>
          <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-5 gap-4">
            {steps.map((s, i) => (
              <figure key={s.label}>
                <img
                  src={s.image}
                  alt={s.label}
                  className={`w-full aspect-square object-cover rounded-lg ${
                    i === steps.length - 1 ? "outline outline-2 outline-offset-2 outline-accent" : ""
                  }`}
                />
                <figcaption className="mt-2">
                  <div className="font-mono text-[11px] font-semibold">
                    {String(i + 1).padStart(2, "0")}
                  </div>
                  <div className="text-xs text-ink-soft leading-snug">{s.label}</div>
                </figcaption>
              </figure>
            ))}
          </div>
        </Card>
      ) : (
        <Unavailable what="The lighting-correction preview" />
      )}
      <Meaning>
        <p>
          HueView measures skin color on the corrected image (the last one, outlined), so a
          shadow on one cheek is less likely to be read as darker skin. The Baseline uses the
          original. Lighting correction can also remove some real color information, which is
          why the study tests its effect instead of assuming it helps.
        </p>
      </Meaning>
    </>
  );
}

function SegmentationPanel({ result }) {
  const hv = result.models.hueview;
  return (
    <>
      <StageHeader id="seg" title="Face regions" />
      {hv.segmentation ? (
        <Card>
          <div className="px-6 pt-5">
            <Happened>
              MediaPipe Face Mesh landmarks outline each region as a convex hull on the
              lighting-corrected face, so the same areas are measured on every photo. An HSV
              filter then removes pixels that are not skin.
            </Happened>
          </div>
          <SegmentationFigures hueview={hv} />
        </Card>
      ) : (
        <Unavailable what="The region segmentation" />
      )}
      <Meaning>
        <p>
          Skin color is not uniform across the face. Instead of one average, HueView reads each
          region separately, so one shadowed or covered area cannot decide the result on its
          own.
        </p>
      </Meaning>
    </>
  );
}

function SkinPanel({ result }) {
  const { baseline, hueview } = result.models;
  const measured = (hueview.regions ?? []).filter(
    (r) => r.used && typeof r.pixels === "number" && typeof r.L === "number",
  );
  const fewest = measured.reduce((m, r) => (!m || r.pixels < m.pixels ? r : m), null);
  const darkest = measured.reduce((m, r) => (!m || r.L < m.L ? r : m), null);
  const excluded = (hueview.regions ?? []).filter((r) => r.used === false);

  return (
    <>
      <StageHeader id="skin" title="Skin filter and color values" />
      <RegionTable regions={hueview.regions ?? []} />
      <ColorValuesCard baseline={baseline} hueview={hueview} />
      <Meaning>
        {fewest && darkest ? (
          <p>
            {measured.length} regions kept enough skin to be measured.{" "}
            <b>{fewest.name}</b> has the fewest skin pixels ({fewest.pixels.toLocaleString()}),
            so it is the least reliable reading in this photo. <b>{darkest.name}</b> is the
            darkest (L* {darkest.L}).
            {excluded.length > 0 &&
              ` ${excluded.map((r) => r.name).join(", ")} ${
                excluded.length === 1 ? "was" : "were"
              } excluded.`}
          </p>
        ) : (
          <p>No region had enough skin pixels to measure.</p>
        )}
        <p className="text-xs text-ink-soft">
          L* = lightness (0 black, 100 white) · a* = redness (+) · b* = yellowness (+).
        </p>
      </Meaning>
    </>
  );
}

function RegionPanel({ result }) {
  const hv = result.models.hueview;
  const voted = (hv.regions ?? []).filter((r) => r.scc);
  const agreeing = voted.filter((r) => r.scc === hv.scc);

  return (
    <>
      <StageHeader id="region" title="What each region predicted" />
      <Card>
        <div className="px-6 pt-5">
          <Happened>
            Each region&apos;s patch went through its own EfficientNetB0 model together with
            that region&apos;s CIELAB values, and was classified into one of the six skin color
            clusters. The percentage is how strongly that model favours its answer.
          </Happened>
        </div>
        <RegionPredictions hueview={hv} />
      </Card>
      {voted.length > 0 && hv.scc && (
        <Meaning>
          <p>
            <b>
              {agreeing.length} of {voted.length} models predicted {hv.scc}
            </b>
            .{" "}
            {agreeing.length === voted.length
              ? "Every region reads the face the same way."
              : `The others predicted ${[
                  ...new Set(voted.filter((r) => r.scc !== hv.scc).map((r) => r.scc)),
                ].join(", ")}.`}
          </p>
        </Meaning>
      )}
    </>
  );
}

function DecisionPanel({ result }) {
  const hv = result.models.hueview;
  const hasCls = (hv.regions ?? []).some((r) => Array.isArray(r.probabilities));

  return (
    <>
      <StageHeader id="decision" title="How the final class is decided" />
      {hasCls ? (
        <Card>
          <DecisionTable hueview={hv} />
        </Card>
      ) : (
        <Unavailable what="The per-region probabilities" />
      )}
      {hv.scc && (
        <Meaning>
          <p>
            HueView&apos;s final class is <b>{sccText(hv)}</b>, the class with the highest mean
            probability across the models ({pct(hv.confidence, 1)}).
          </p>
        </Meaning>
      )}
    </>
  );
}

function UndertonePanel({ result }) {
  const { baseline, hueview } = result.models;
  return (
    <>
      <StageHeader id="under" title="Undertone" />
      <UndertoneCard baseline={baseline} hueview={hueview} />
    </>
  );
}

function ComparePanel({ result }) {
  const { baseline, hueview } = result.models;
  const bothPredicted = Boolean(baseline.scc && hueview.scc);
  const agree = bothPredicted && baseline.scc === hueview.scc;

  return (
    <>
      <StageHeader id="compare" title="Baseline vs HueView" />
      <Card className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-blush">
              {["For this photo", "Baseline", "HueView"].map((h) => (
                <th
                  key={h}
                  className="px-5 py-3 text-left font-mono text-[10px] tracking-widest uppercase text-accent font-normal"
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
            <Row k="Predicted class" a={sccText(baseline)} b={sccText(hueview)} highlight />
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
      {bothPredicted && (
        <Meaning>
          <p>
            {agree
              ? `For this photo, both pipelines agree on ${sccText(hueview)}.`
              : `For this photo, the pipelines disagree: ${sccText(baseline)} against ${sccText(hueview)}.`}{" "}
            They see different inputs (the whole photo vs corrected skin regions), so they can
            disagree on individual faces.
          </p>
        </Meaning>
      )}
    </>
  );
}

function Row({ k, a, b, highlight }) {
  return (
    <tr className="border-t border-line-soft">
      <td className="px-5 py-3 text-ink-soft">{k}</td>
      <td className={`px-5 py-3 ${highlight ? "font-semibold" : ""}`}>{a}</td>
      <td className={`px-5 py-3 ${highlight ? "font-semibold text-accent" : ""}`}>{b}</td>
    </tr>
  );
}
