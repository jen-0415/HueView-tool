import Card, { PanelHead } from "./Card";
import UndertoneBar from "./UndertoneBar";
import { UNDERTONE_RULES } from "../constants";

// The two pipelines use DIFFERENT undertone rules, not one rule on two inputs:
//   Baseline — normalized RGB ratios, thresholded on the b ratio.
//   ChroLumi  — hue angle on the regional CIELAB means.
// Each side names its own rule so the comparison is not read as like-for-like.
export default function UndertoneCard({ baseline, hueview, arriving = false }) {
  const agree = baseline.undertone.label === hueview.undertone.label;

  return (
    <Card>
      <PanelHead
        title="Undertone"
        note="Each pipeline finds undertone its own way, so compare the answers, not the methods."
        right={
          <span className="shrink-0 rounded-full bg-warm text-warm-ink text-data px-2.5 py-0.5">
            experimental
          </span>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-line-soft">
        <Side model={baseline} arriving={arriving} />
        <Side model={hueview} primary arriving={arriving} />
      </div>

      <p className="px-6 py-4 border-t border-line-soft text-fine text-ink-mid leading-relaxed">
        {agree
          ? "Both pipelines give the same undertone."
          : `The pipelines disagree: ${baseline.undertone.label} against ${hueview.undertone.label}.`}{" "}
        Undertone hasn&apos;t been checked against confirmed labels and is outside the main scope
        of the study.
      </p>
    </Card>
  );
}

function Side({ model, primary, arriving }) {
  const u = model.undertone;
  const rule = UNDERTONE_RULES[u.method];

  return (
    <div className="px-6 py-5">
      <div className="font-semibold text-fine">{model.name}</div>
      <p className="text-fine text-ink-mid mt-0.5">{rule?.title}</p>

      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mt-3">
        <span
          className={`font-display text-title font-semibold ${
            primary ? "text-accent" : "text-ink"
          }`}
        >
          {u.label}
        </span>
        <span className="font-mono text-fine text-ink-mid">
          {u.method === "rgb_ratio"
            ? `blue share ${u.b_ratio}`
            : `average hue ${u.hue_angle_deg ?? "—"}°`}
        </span>
      </div>

      {/* Baseline: the three normalized ratios and the threshold bands. */}
      {u.method === "rgb_ratio" && (
        <div className="mt-5">
          <div className="flex gap-5 font-mono text-fine">
            <span className="text-ink-mid">
              r <span className="text-ink">{u.ratios.r}</span>
            </span>
            <span className="text-ink-mid">
              g <span className="text-ink">{u.ratios.g}</span>
            </span>
            <span className="text-accent">
              b <span className="font-semibold">{u.ratios.b}</span>
            </span>
          </div>

          <ul className="mt-3 flex flex-col gap-1">
            {rule.bands.map((band) => {
              const hit = band.label === u.label;
              return (
                <li
                  key={band.label}
                  className={`flex justify-between gap-4 font-mono text-data rounded px-2.5 py-1.5 ${
                    hit
                      ? "bg-blush-soft text-accent font-semibold border border-line"
                      : "text-ink-mid"
                  }`}
                >
                  <span>{band.test}</span>
                  <span>{band.label}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* ChroLumi: the spread across the selected regions. */}
      {u.method === "hue_angle" && (
        <div className="mt-5">
          <div className="text-data text-ink-mid mb-2">Across face areas</div>
          <UndertoneBar distribution={u.distribution} arriving={arriving} />
          <p className="text-fine text-ink-mid mt-2.5">
            How many of the face areas fall under each undertone.
          </p>
        </div>
      )}
    </div>
  );
}
