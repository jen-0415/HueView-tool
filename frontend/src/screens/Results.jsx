import { useState } from "react";
import Card from "../components/Card";
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

// One photo's result, shown the same way to every reader (the user/researcher
// switch only controls the Evaluation Results tab -- see view.js).
//
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
            gave no result because its model isn&apos;t loaded, so the two can&apos;t be compared
            for this photo.
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5 items-start">
        <ModelCard model={baseline} />
        <ModelCard model={hueview} primary />
      </div>

      <SccScaleCard predicted={hueview.scc} />

      <Meaning>
        <p>
          The Baseline looks at the whole original photo, including hair, background and
          shadows. HueView fixes the lighting first, then looks only at skin in separate
          parts of the face. Because they look at different things, they can give different
          answers.
        </p>
        <p>
          The undertone is <b>experimental</b>. It has not been checked against confirmed
          labels, so treat it as a rough guide.
        </p>
        <div className="flex flex-wrap gap-2 pt-2 items-center">
          <span className="text-[13px] text-ink-soft">See where the result comes from:</span>
          <Chip onClick={() => pick("ssr")}>03 Lighting correction →</Chip>
          <Chip onClick={() => pick("region")}>06 Region results →</Chip>
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
          {d.landmarks_found} of 5 key points were found. Every later stage works on the
          cropped face shown here.
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
      <StageHeader id="illum" title="Lighting level" />
      <div className="grid grid-cols-1 md:grid-cols-[320px_1fr] gap-5 items-start">
        <IlluminationCard illumination={il} />
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

function SsrPanel({ ssrPreview }) {
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
          HueView reads skin color from the corrected face (the last image, outlined), so a
          shadow on one cheek is less likely to be mistaken for darker skin. The Baseline uses
          the original photo. The correction can also remove some real color, which is why the
          study tests whether it helps instead of assuming it does.
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
          Skin color isn&apos;t the same all over the face. HueView reads each area separately
          instead of taking one average, so a single shadowed or covered area can&apos;t decide
          the result on its own.
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
      <StageHeader id="skin" title="Skin pixels and color values" />
      <RegionTable regions={hueview.regions ?? []} />
      <ColorValuesCard baseline={baseline} hueview={hueview} />
      <Meaning>
        {fewest && darkest ? (
          <p>
            {measured.length} areas had enough skin to measure.{" "}
            <b>{fewest.name}</b> had the least skin ({fewest.pixels.toLocaleString()} pixels),
            so its reading is the least reliable in this photo. <b>{darkest.name}</b> is the
            darkest area (lightness {darkest.L}).
            {excluded.length > 0 &&
              ` ${excluded.map((r) => r.name).join(", ")} ${
                excluded.length === 1 ? "was" : "were"
              } excluded.`}
          </p>
        ) : (
          <p>No area had enough skin to measure.</p>
        )}
        <p className="text-xs text-ink-soft">
          L* = lightness (0 is black, 100 is white) · a* = how red · b* = how yellow.
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
            HueView&apos;s final answer is <b>{sccText(hv)}</b>, the class picked by the most of
            its six models — the five face regions and the full face
            {hv.votes?.[hv.scc] != null ? ` (${hv.votes[hv.scc]} of 6)` : ""}
            {hv.tied ? ". It was a tie, so the class with the higher average score won" : ""}.
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
              k="What it looks at"
              a="The whole cropped photo and its average color"
              b="Skin from five face areas plus the whole face, after lighting correction"
            />
            <Row k="Predicted class" a={sccText(baseline)} b={sccText(hueview)} highlight />
            <Row k="Color space" a="RGB" b="CIELAB" />
            <Row
              k="How undertone is found"
              a="How much blue is in the average color"
              b="The color direction (hue) of each face area"
            />
            <Row
              k="Undertone"
              a={`${baseline.undertone.label} (blue share ${baseline.undertone.b_ratio})`}
              b={`${hueview.undertone.label ?? "—"} (average hue ${hueview.undertone.hue_angle_deg ?? "—"}°)`}
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
    <tr className="border-t border-line-soft">
      <td className="px-5 py-3 text-ink-soft">{k}</td>
      <td className={`px-5 py-3 ${highlight ? "font-semibold" : ""}`}>{a}</td>
      <td className={`px-5 py-3 ${highlight ? "font-semibold text-accent" : ""}`}>{b}</td>
    </tr>
  );
}
