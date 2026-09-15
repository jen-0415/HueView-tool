import Card from "./Card";
import SccStrip from "./SccStrip";
import { sccLabel } from "../constants";

// One component renders BOTH models. Same structure = same visual weight,
// and it makes it impossible to accidentally flatter one pipeline.
// Color values are deliberately not here — see ColorValuesCard.
export default function ModelCard({ model, primary = false }) {
  const tint = primary ? "text-accent" : "text-ink-soft";
  const head = primary ? "bg-blush-soft" : "bg-neutraltone";

  return (
    <Card className={primary ? "border-accent" : ""}>
      <div className={`${head} border-b border-line-soft rounded-t-2xl px-6 py-4 flex justify-between items-start`}>
        <div>
          <div className="font-semibold text-[17px]">{model.name}</div>
          <div className="text-[13px] text-ink-soft">{model.method}</div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[10px] text-ink-soft">ckpt {model.checkpoint}</div>
          {model.placeholder && (
            <div className="mt-1 inline-block rounded-full bg-warm text-warm-ink font-mono text-[9px] px-2 py-1">
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
        />

        <div className={`font-display text-[26px] mt-6 ${tint}`}>
          {model.scc} · {sccLabel(model.scc)}
          {model.undertone ? `, ${model.undertone.label}` : ""}
        </div>
      </div>
    </Card>
  );
}