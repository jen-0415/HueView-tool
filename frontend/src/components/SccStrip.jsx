import { SCC } from "../constants";

// The six swatches, with the predicted class raised and outlined.
//
// When probabilities/predicted are null -- no trained weights loaded yet,
// which is the real current state for both models -- this renders all six
// neutrally with a plain caption instead of guessing at percentages. That's
// a real state to show honestly, not a loading glitch to paper over with
// fake numbers.
export default function SccStrip({ predicted, probabilities, tint = "accent" }) {
  const outline = tint === "accent" ? "outline-accent" : "outline-ink-soft";
  const text = tint === "accent" ? "text-accent" : "text-ink-soft";
  const hasPrediction = predicted != null && Array.isArray(probabilities);

  return (
    <div>
      <div className="flex gap-2">
        {SCC.map((s, i) => {
          const on = hasPrediction && s.id === predicted;
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
                {hasPrediction ? `${Math.round(probabilities[i] * 100)}%` : "\u2013"}
              </div>
            </div>
          );
        })}
      </div>
      {!hasPrediction && (
        <div className="font-mono text-[10px] text-ink-soft mt-2">
          No prediction yet -- awaiting trained model weights.
        </div>
      )}
    </div>
  );
}