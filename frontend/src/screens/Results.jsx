import { useState } from "react";
import Card, { Verdict, Reading } from "../components/Card";
import ModelCard from "../components/ModelCard";
import ColorValuesCard from "../components/ColorValuesCard";
import IlluminationCard from "../components/IlluminationCard";
import UndertoneCard from "../components/UndertoneCard";
import RegionTable from "../components/RegionTable";
import ResultsSidebar from "../components/ResultsSidebar";
import SccScaleCard from "../components/SccScaleCard";
import {
  SegmentationFigures,
  RegionPredictions,
  DecisionTable,
} from "../components/RegionalSegmentationCard";
import { sccLabel } from "../constants";
import { STEPS, ORDER } from "../stages";
import { useArrival } from "../motion";

// One photo's result, shown the same way to every reader (the user/researcher
// switch only controls the Evaluation Results tab -- see view.js).
//
// One stage at a time, picked from the sidebar. Each stage panel shows what
// that step did ("What happened") and a plain-language reading of this
// photo's values ("What this means"), written from the real result.
export default function Results({ preview, result, ssrPreview, onReset }) {
  const [stage, setStage] = useState("summary");
  const idx = ORDER.findIndex((o) => o.id === stage);
  // One orchestrated moment per analysis: held here, not asked for by each
  // panel, so switching stages never replays it.
  const arriving = useArrival();

  const pick = (id) => {
    setStage(id);
    window.scrollTo({ top: 0 });
  };

  const props = { result, ssrPreview, preview, pick, arriving };

  return (
    <main className="max-w-[1360px] mx-auto px-4 sm:px-8 py-7 flex flex-col lg:flex-row gap-7 items-start">
      <ResultsSidebar
        stage={stage}
        onPick={pick}
        preview={result.image.preview || preview}
        onReset={onReset}
      />

      <section className="flex-1 min-w-0 w-full flex flex-col gap-5">
        {/* Keyed on the stage so each panel crossfades in on switch. */}
        <div key={stage} className="panel-in flex flex-col gap-5">
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
        </div>

        <div className="flex justify-between gap-3 flex-wrap border-t border-line pt-4">
          <div>
            {idx > 0 && (
              <button
                onClick={() => pick(ORDER[idx - 1].id)}
                className="rounded-full bg-white border border-line text-accent text-fine px-5 min-h-[44px] hover:border-accent transition-colors duration-200"
              >
                <span className="text-ink-mid">Back to</span> {ORDER[idx - 1].label}
              </button>
            )}
          </div>
          <div>
            {idx < ORDER.length - 1 && (
              <button
                onClick={() => pick(ORDER[idx + 1].id)}
                className="rounded-full bg-accent text-white text-fine font-semibold px-5 min-h-[44px] hover:bg-accent-light transition-colors duration-200"
              >
                Next: {ORDER[idx + 1].label}
              </button>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

/* ---------- shared bits ---------- */

const sccText = (m) => (m.scc ? `${m.scc} ${sccLabel(m.scc)}` : "No prediction");

// Which pipelines a stage belongs to is a property of the stage, so it is
// shown as a state, not buried in a sentence. The sidebar's rail uses the
// same three values.
const SCOPE_STYLE = {
  "shared by both pipelines": "bg-neutraltone text-ink-mid",
  "ChroLumi only": "bg-blush-soft text-accent border border-line",
  experimental: "bg-warm text-warm-ink",
};

function StageHeader({ id, title }) {
  const step = STEPS.find((s) => s.id === id);
  return (
    <div>
      <div className="flex items-center gap-3 flex-wrap">
        <span className="font-mono text-data text-ink-mid">
          {step ? `Stage ${step.n} of ${String(STEPS.length).padStart(2, "0")}` : "Stage"}
        </span>
        {step && (
          <span
            className={`rounded-full px-2.5 py-0.5 text-data ${
              SCOPE_STYLE[step.scope] ?? "bg-neutraltone text-ink-mid"
            }`}
          >
            {step.scope}
          </span>
        )}
      </div>
      <h2 className="font-display text-head font-semibold mt-2">{title}</h2>
    </div>
  );
}

// What this step did. The paragraph is self-evidently an explanation, so it no
// longer carries a label above it.
function Happened({ children }) {
  return <p className="text-base leading-relaxed text-ink-mid mb-5">{children}</p>;
}

const Meaning = Reading;

function Unavailable({ what }) {
  return (
    <Card className="px-6 py-5 text-base text-ink-mid">
      {what} isn&apos;t available for this result.
    </Card>
  );
}

function Chip({ onClick, children }) {
  return (
    <button
      onClick={onClick}
      className="rounded-full bg-white border border-line text-accent text-fine px-4 min-h-[36px] hover:border-accent transition-colors duration-200"
    >
      {children}
    </button>
  );
}

/* ---------- panels ---------- */

function SummaryPanel({ result, pick, arriving }) {
  const { baseline, hueview } = result.models;
  // A null SCC means that model's weights aren't loaded -- never call two
  // missing predictions an agreement.
  const bothPredicted = Boolean(baseline.scc && hueview.scc);
  const agree = bothPredicted && baseline.scc === hueview.scc;
  const missing = [baseline, hueview].filter((m) => !m.scc).map((m) => m.name);

  return (
    <>
      <div>
        <h2 className="font-display text-head font-semibold">Analysis results</h2>
        <dl className="flex flex-wrap gap-x-7 gap-y-1 mt-3 text-data font-mono text-ink-mid">
          <div>
            <dt className="inline">Architecture </dt>
            <dd className="inline text-ink">EfficientNetB0</dd>
          </div>
          <div>
            <dt className="inline">Weighting </dt>
            <dd className="inline text-ink">STW</dd>
          </div>
          <div>
            <dt className="inline">Baseline ckpt </dt>
            <dd className="inline text-ink">{baseline.checkpoint}</dd>
          </div>
          <div>
            <dt className="inline">ChroLumi ckpt </dt>
            <dd className="inline text-ink">{hueview.checkpoint}</dd>
          </div>
        </dl>
      </div>

      {/* The headline finding. Do not bury this in a table. */}
      {bothPredicted ? (
        <Verdict
          tone={agree ? "agree" : "disagree"}
          headline={agree ? "Both pipelines agree" : "The pipelines disagree"}
        >
          {agree
            ? `Both classified this face as ${sccText(baseline)}.`
            : `Baseline read this face as ${sccText(baseline)}; ChroLumi read it as ${sccText(hueview)}.`}
        </Verdict>
      ) : (
        <Verdict tone="caution" headline="Comparison unavailable">
          {missing.join(" and ")} gave no result because its model isn&apos;t loaded, so the two
          can&apos;t be compared for this photo.
        </Verdict>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5 items-start">
        <ModelCard model={baseline} arriving={arriving} />
        <ModelCard model={hueview} primary arriving={arriving} />
      </div>

      <SccScaleCard predicted={hueview.scc} />

      <Meaning>
        <p>
          The Baseline looks at the whole original photo, including hair, background and
          shadows. ChroLumi fixes the lighting first, then looks only at skin in separate
          parts of the face. Because they look at different things, they can give different
          answers.
        </p>
        <p>
          The undertone is <b>experimental</b>. It has not been checked against confirmed
          labels, so treat it as a rough guide.
        </p>
        <div className="flex flex-wrap gap-2 pt-2 items-center">
          <span className="text-fine text-ink-mid">See where the result comes from:</span>
          <Chip onClick={() => pick("ssr")}>03 Lighting correction</Chip>
          <Chip onClick={() => pick("region")}>06 Region results</Chip>
          <Chip onClick={() => pick("decision")}>07 Final decision</Chip>
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
          className="w-[224px] h-[224px] object-cover rounded-readout border border-rule"
        />
        <div className="flex-[1_1_300px] min-w-0">
          <Happened>
            The face was found in your photo, along with five key points: both eyes, the tip
            of the nose and both corners of the mouth. The face was then cropped and resized to{" "}
            {result.image.width} × {result.image.height} pixels, the size both models need.
          </Happened>
          <Value k="Key points found" v={`${d.landmarks_found} / 5`}>
            Used to line up and crop the face.
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
          {d.landmarks_found} of 5 key points were found.
        </p>
      </Meaning>
    </>
  );
}

function Value({ k, v, children }) {
  return (
    <div className="grid grid-cols-[170px_minmax(0,1fr)] gap-x-4 gap-y-1 py-3.5 border-t border-line-soft">
      <span className="text-fine text-ink-mid">{k}</span>
      <span className="font-mono text-lead">{v}</span>
      <span className="col-start-2 text-fine text-ink-mid leading-snug">{children}</span>
    </div>
  );
}

function IlluminationPanel({ result, arriving }) {
  const il = result.illumination;
  return (
    <>
      <StageHeader id="illum" title="Lighting level" />
      <div className="grid grid-cols-1 md:grid-cols-[320px_1fr] gap-5 items-start">
        <IlluminationCard illumination={il} arriving={arriving} />
        <Card className="p-6">
          <Happened>
            We measured how bright the photo is and placed it in one of three lighting groups,
            based on all the photos in our dataset. This helps us check how well the models do
            under different lighting. <b>It does not change the result.</b>
          </Happened>
        </Card>
      </div>
      <Meaning>
        <p>
          This photo falls in the <b>{il.bin}</b> lighting group (brightness {il.value} on a{" "}
          {il.scale[0]}–{il.scale[1]} scale).
        </p>
      </Meaning>
    </>
  );
}

function SsrPanel({ ssrPreview, arriving }) {
  // The backend omits `steps` if its step preview stops matching the real SSR
  // output; fall back to just before/after.
  const steps = ssrPreview
    ? ssrPreview.steps ?? [
        { label: "Your cropped face", image: ssrPreview.original },
        { label: "Corrected face", image: ssrPreview.ssr },
      ]
    : null;

  return (
    <>
      <StageHeader id="ssr" title="Lighting correction" />
      {steps ? (
        <Card className="p-6">
          <Happened>
            A heavy blur of the face gives a rough picture of how the light falls on it. That
            lighting is then taken out, which evens out shadows and bright spots and leaves
            something closer to the skin&apos;s real color.
          </Happened>
          <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-5 gap-4">
            {steps.map((s, i) => (
              <figure
                key={s.label}
                className={arriving ? "panel-in" : ""}
                style={arriving ? { animationDelay: `${i * 70}ms` } : undefined}
              >
                <img
                  src={s.image}
                  alt={s.label}
                  className={`w-full aspect-square object-cover rounded-readout ${
                    i === steps.length - 1
                      ? "outline outline-2 outline-offset-2 outline-accent"
                      : "border border-rule"
                  }`}
                />
                <figcaption className="mt-2.5">
                  <div className="font-mono text-data text-ink-mid">
                    {String(i + 1).padStart(2, "0")}
                  </div>
                  <div className="text-fine text-ink-mid leading-snug mt-0.5">{s.label}</div>
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
          ChroLumi reads skin color from the corrected face (the last image, outlined), so a
          shadow on one cheek is less likely to be mistaken for darker skin. The Baseline uses
          the original photo.
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
              Points mapped across the face are used to outline the same areas on every photo
              (forehead, cheeks, nose and chin). A skin filter then removes anything that
              isn&apos;t skin, such as eyebrows, hair or shadows.
            </Happened>
          </div>
          <SegmentationFigures hueview={hv} />
        </Card>
      ) : (
        <Unavailable what="The region segmentation" />
      )}
      <Meaning>
        <p>
          Skin color isn&apos;t the same all over the face. ChroLumi reads each area separately
          instead of taking one average, so a single shadowed or covered area can&apos;t decide
          the result on its own.
        </p>
      </Meaning>
    </>
  );
}

function SkinPanel({ result, arriving }) {
  const { baseline, hueview } = result.models;
  const measured = (hueview.regions ?? []).filter(
    (r) => r.used && typeof r.pixels === "number" && typeof r.L === "number",
  );
  const fewest = measured.reduce((m, r) => (!m || r.pixels < m.pixels ? r : m), null);
  const darkest = measured.reduce((m, r) => (!m || r.L < m.L ? r : m), null);
  // Full Face is the five areas' average, never a skipped area.
  const excluded = (hueview.regions ?? []).filter(
    (r) => r.used === false && r.region_key !== "full_face",
  );

  return (
    <>
      <StageHeader id="skin" title="Skin pixels and color values" />
      <RegionTable regions={hueview.regions ?? []} />
      <ColorValuesCard baseline={baseline} hueview={hueview} arriving={arriving} />
      <Meaning>
        {fewest && darkest ? (
          <p>
            Skin color was measured in {measured.length} of the 5 face{" "}
            {measured.length === 1 ? "area" : "areas"}. The <b>{fewest.name}</b> reading is based on
            the fewest skin pixels ({fewest.pixels.toLocaleString()}), so it is the least certain.
            The <b>{darkest.name}</b> is the darkest area, with the lowest lightness (L* {darkest.L}).
            {excluded.length > 0 &&
              ` ${excluded.map((r) => r.name).join(", ")} had too little skin to measure, so ${
                excluded.length === 1 ? "its value was" : "their values were"
              } filled in from the other areas.`}
          </p>
        ) : (
          <p>No area had enough skin to measure.</p>
        )}
        <p className="text-fine text-ink-mid">
          L* is lightness, from 0 (black) to 100 (white). a* runs from green (−) to red (+),
          and b* from blue (−) to yellow (+).
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
            Each face area has its own model. It looks at that area&apos;s image and color values
            and picks one of the six skin color classes. The percentage shows how strongly the
            model leans toward its pick.
          </Happened>
        </div>
        <RegionPredictions hueview={hv} />
      </Card>
      {voted.length > 0 && hv.scc && (
        <Meaning>
          <p>
            <b>
              {agreeing.length} of {voted.length} models picked {hv.scc}
            </b>
            .{" "}
            {agreeing.length === voted.length
              ? "Every area gave the same answer."
              : `The others picked ${[
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
            ChroLumi&apos;s final answer is <b>{sccText(hv)}</b>, the class picked by the most of
            its six models — the five face regions and the full face
            {hv.votes?.[hv.scc] != null ? ` (${hv.votes[hv.scc]} of 6)` : ""}
            {hv.tied ? ". It was a tie, so the class with the higher average score won" : ""}.
          </p>
        </Meaning>
      )}
    </>
  );
}

function UndertonePanel({ result, arriving }) {
  const { baseline, hueview } = result.models;
  return (
    <>
      <StageHeader id="under" title="Undertone" />
      <UndertoneCard baseline={baseline} hueview={hueview} arriving={arriving} />
    </>
  );
}

function ComparePanel({ result }) {
  const { baseline, hueview } = result.models;
  const bothPredicted = Boolean(baseline.scc && hueview.scc);
  const agree = bothPredicted && baseline.scc === hueview.scc;

  return (
    <>
      <StageHeader id="compare" title="Baseline vs ChroLumi" />
      <Card>
        <div className="overflow-x-auto">
        <table className="w-full text-base min-w-[560px]">
          <thead>
            <tr className="bg-blush">
              {["For this photo", "Baseline", "ChroLumi"].map((h) => (
                <th
                  key={h}
                  className="sticky top-0 bg-blush px-5 py-3 text-left text-fine font-medium text-accent"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <Row
              k="What it looks at"
              a="The whole cropped photo and its average color"
              b="Skin from five face areas plus the whole face, after lighting correction"
            />
            <Row k="Color space" a="RGB" b="CIELAB" />
            <Row
              k="Predicted skin tone class"
              a={sccText(baseline)}
              b={sccText(hueview)}
              highlight
            />
            <Row
              k="Undertone"
              a={`${baseline.undertone.label} (blue share ${baseline.undertone.b_ratio})`}
              b={`${hueview.undertone.label ?? "—"} (average hue ${hueview.undertone.hue_angle_deg ?? "—"}°)`}
            />
          </tbody>
        </table>
        </div>
        <p className="px-5 py-3.5 border-t border-line-soft text-fine text-ink-mid">
          How undertone is found: the Baseline uses how much blue is in the average color; ChroLumi
          uses the color direction (hue) of each face area.
        </p>
      </Card>
      {bothPredicted && (
        <Meaning>
          <p>
            {agree
              ? `For this photo, both pipelines agree on ${sccText(hueview)}.`
              : `For this photo, the pipelines disagree: ${sccText(baseline)} against ${sccText(hueview)}.`}{" "}
            They look at different things (the whole photo vs corrected skin areas), so they can
            disagree on some faces.
          </p>
        </Meaning>
      )}
    </>
  );
}

function Row({ k, a, b, highlight }) {
  return (
    <tr className={`border-t border-rule/60 ${highlight ? "bg-blush-soft/50" : ""}`}>
      <td className="px-5 py-3.5 text-fine text-ink-mid align-top">{k}</td>
      <td className={`px-5 py-3.5 align-top ${highlight ? "font-semibold" : ""}`}>{a}</td>
      <td className={`px-5 py-3.5 align-top ${highlight ? "font-semibold text-accent" : ""}`}>
        {b}
      </td>
    </tr>
  );
}
