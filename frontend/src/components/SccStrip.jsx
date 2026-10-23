import { SCC } from "../constants";
import { measureCls, stagger } from "../motion";

// The six swatches, with the predicted class raised and outlined, and each
// class's probability as a bar under it.
//
// The bar is the measured geometry: it is what animates on arrival, scaling
// from zero to the real probability. The percentage above it is plain text and
// is correct from the first frame.
//
// When probabilities/predicted are null -- no trained weights loaded yet,
// which is the real current state for both models -- this renders all six
// neutrally with a plain caption instead of guessing at percentages. That's
// a real state to show honestly, not a loading glitch to paper over with
// fake numbers.
export default function SccStrip({ predicted, probabilities, tint = "accent", arriving = false }) {
  const outline = tint === "accent" ? "outline-accent" : "outline-ink-soft";
  const fill = tint === "accent" ? "bg-accent" : "bg-ink-soft";
  const hasPrediction = predicted != null && Array.isArray(probabilities);

  return (
    <div>
      <div className="flex items-end gap-2">
        {SCC.map((s, i) => {
          const on = hasPrediction && s.id === predicted;
          const p = hasPrediction ? probabilities[i] : null;
          return (
            <div key={s.id} className="flex-1 min-w-0">
              <div
                className={`w-full rounded-lg transition-[height] duration-300 ease-settle ${
                  on ? `h-14 outline outline-2 outline-offset-2 ${outline}` : "h-9"
                }`}
                style={{ background: s.hex }}
                role="img"
                aria-label={`${s.id} ${s.label}`}
              />
              <div
                className={`font-mono text-data mt-2.5 ${
                  on ? "text-accent font-semibold" : "text-ink-mid"
                }`}
              >
                {s.id.replace("SCC-", "")}
              </div>
              {hasPrediction ? (
                <>
                  <div className="h-1.5 rounded-full bg-blush mt-1.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${on ? fill : "bg-line"} ${measureCls(
                        arriving,
                      )}`}
                      style={{ width: `${Math.round(p * 100)}%`, ...stagger(i) }}
                    />
                  </div>
                  <div
                    className={`font-mono text-data mt-1 ${on ? "text-ink" : "text-ink-mid"}`}
                  >
                    {Math.round(p * 100)}%
                  </div>
                </>
              ) : (
                <div className="font-mono text-data text-ink-mid mt-1.5">&mdash;</div>
              )}
            </div>
          );
        })}
      </div>
      {!hasPrediction && (
        <p className="font-mono text-data text-ink-mid mt-3">
          No prediction yet &mdash; awaiting trained model weights.
        </p>
      )}
    </div>
  );
}
