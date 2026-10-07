import Card from "./Card";
import UndertoneBar from "./UndertoneBar";
import { UNDERTONE_RULES } from "../constants";

// The two pipelines use DIFFERENT undertone rules, not one rule on two inputs:
//   Baseline — normalized RGB ratios, thresholded on the b ratio.
//   HueView  — hue angle on the regional CIELAB means.
// Each side names its own rule so the comparison is not read as like-for-like.
export default function UndertoneCard({ baseline, hueview }) {
    const agree = baseline.undertone.label === hueview.undertone.label;

    return (
        <Card>
            <div className="px-6 py-4 border-b border-line-soft">
                <div className="font-semibold">Undertone descriptor</div>
                <div className="text-xs text-ink-soft">
                    Each pipeline uses its own rule — compare the outputs, not the methods.
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-line-soft">
                <Side model={baseline} />
                <Side model={hueview} primary />
            </div>

            <p className="px-6 py-4 border-t border-line-soft text-[13px] text-ink-soft leading-relaxed">
                {agree
                    ? "Both pipelines land in the same undertone band."
                    : `The pipelines disagree: ${baseline.undertone.label} against ${hueview.undertone.label}.`}{" "}
                Undertone is an exploratory chromatic descriptor, not a validated
                classification, and it is not covered by the study's hypotheses.
            </p>
        </Card>
    );
}

function Side({ model, primary }) {
    const u = model.undertone;
    const rule = UNDERTONE_RULES[u.method];

    return (
        <div className="px-6 py-5">
            <div className="font-mono text-[10px] tracking-widest text-ink-soft">
                {model.name.toUpperCase()}
            </div>

            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mt-2">
                <span className={`font-display text-[26px] ${primary ? "text-accent" : "text-ink"}`}>
                    {u.label}
                </span>
                <span className="font-mono text-[12px] text-ink-soft">
                    {u.method === "rgb_ratio"
                        ? `b ratio ${u.b_ratio}`
                        : `mean regional hue ${u.hue_angle_deg ?? "—"}°`}
                </span>
            </div>

            {/* Baseline: the three normalized ratios and the threshold bands. */}
            {u.method === "rgb_ratio" && (
                <div className="mt-4">
                    <div className="flex gap-4 font-mono text-[12px]">
                        <span>r {u.ratios.r}</span>
                        <span>g {u.ratios.g}</span>
                        <span className="text-accent">b {u.ratios.b}</span>
                    </div>

                    <ul className="mt-3 flex flex-col gap-1">
                        {rule.bands.map((band) => {
                            const hit = band.label === u.label;
                            return (
                                <li
                                    key={band.label}
                                    className={`flex justify-between font-mono text-[11px] rounded px-2 py-1 ${hit ? "bg-blush-soft text-accent" : "text-ink-soft"
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

            {/* HueView: the spread across the selected regions. */}
            {u.method === "hue_angle" && (
                <div className="mt-4">
                    <div className="font-mono text-[10px] tracking-widest text-ink-soft mb-2">
                        REGIONAL DISTRIBUTION
                    </div>
                    <UndertoneBar distribution={u.distribution} />
                    <p className="text-[11px] text-ink-soft mt-2">
                        Share of the selected regions falling in each band.
                    </p>
                </div>
            )}
        </div>
    );
}