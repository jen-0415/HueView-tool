import { SCC } from "../constants";

// The six swatches, with the predicted class raised and outlined.
export default function SccStrip({ predicted, probabilities, tint = "accent" }) {
  const outline = tint === "accent" ? "outline-accent" : "outline-ink-soft";
  const text = tint === "accent" ? "text-accent" : "text-ink-soft";

  return (
    <div className="flex gap-2">
      {SCC.map((s, i) => {
        const on = s.id === predicted;
        return (
          <div key={s.id} className="flex-1">
            <div
              className={`w-full rounded-lg transition-all ${
                on ? `h-12 outline outline-2 outline-offset-2 ${outline}` : "h-8"
              }`}
              style={{ background: s.hex }}
            />
            <div className={`font-mono text-[9px] mt-2 ${on ? text : "text-ink-soft"}`}>
              {s.id}
            </div>
            <div className="font-mono text-[9px] text-ink-soft">
              {Math.round(probabilities[i] * 100)}%
            </div>
          </div>
        );
      })}
    </div>
  );
}
