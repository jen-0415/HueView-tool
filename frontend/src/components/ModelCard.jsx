import Card from "./Card";
import SccStrip from "./SccStrip";
import { SCC, sccLabel, sccMst } from "../constants";

const SCC_HEX = Object.fromEntries(SCC.map((s) => [s.id, s.hex]));

// One component renders BOTH models. Same structure = same visual weight,
// and it makes it impossible to accidentally flatter one pipeline. `primary`
// only tints; it never changes the layout or what is shown.
// Color values are deliberately not here -- see ColorValuesCard.
//
// This is the readout tier: a value you are meant to read off, with a rule on
// its measured edge.
export default function ModelCard({ model, primary = false, arriving = false }) {
  const undertoneLabel = model.undertone?.label;
  const swatch = model.scc ? SCC_HEX[model.scc] : null;

  return (
    <Card tier="readout" className="overflow-hidden">
      <div
        className={`flex items-start justify-between gap-4 px-6 py-3.5 border-b ${
          primary ? "bg-blush-soft border-line" : "bg-neutraltone border-line-soft"
        }`}
      >
        <div className="font-semibold text-lead">{model.name}</div>
        <div className="text-right">
          <div className="font-mono text-data text-ink-mid">ckpt {model.checkpoint}</div>
          {model.placeholder && (
            <div className="mt-1 inline-block rounded-full bg-warm text-warm-ink font-mono text-data px-2 py-0.5">
              placeholder weights
            </div>
          )}
        </div>
      </div>

      <div className="px-6 py-5">
        <SccStrip
          predicted={model.scc}
          probabilities={model.probabilities}
          tint={primary ? "accent" : "muted"}
          arriving={arriving}
        />

        <div className="mt-6 flex items-start gap-3.5">
          {swatch && (
            <span
              className="mt-1 w-10 h-10 shrink-0 rounded-lg border border-rule"
              style={{ background: swatch }}
              aria-hidden="true"
            />
          )}
          <div className="min-w-0">
            {model.scc ? (
              <>
                <div
                  className={`font-display text-title font-semibold ${
                    primary ? "text-accent" : "text-ink"
                  }`}
                >
                  {model.scc} {sccLabel(model.scc)}
                </div>
                {/* The class name alone says nothing about where it sits; the
                    MST steps it covers do. Full definition in SccScaleCard. */}
                <div className="font-mono text-data text-ink-mid mt-1.5">
                  {sccMst(model.scc)} on the Monk Skin Tone scale
                </div>
                {undertoneLabel && (
                  <div className="text-fine text-ink-mid mt-1">
                    {undertoneLabel} undertone
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="font-display text-title font-semibold text-ink-mid">
                  Awaiting trained model
                </div>
                {/* Undertone is rule-based, not learned -- it's real right now
                    even though SCC classification is still pending weights.
                    Worth showing rather than hiding behind the same "pending"
                    state as the classification. */}
                {undertoneLabel && (
                  <div className="text-fine text-ink-mid mt-1">
                    {undertoneLabel} undertone, measured from this photo
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
