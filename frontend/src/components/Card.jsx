// The surface tiers.
//
// Everything used to be one shape -- a white rounded-2xl box -- so the
// headline finding, a reference table and a status note all carried the same
// visual weight. Three tiers now, and which one a thing gets says what kind of
// thing it is:
//
//   panel    reference material: tables, scales, figures, explanation
//   readout  a value you are meant to read off, lifted and ruled on its
//            measured edge
//   verdict  the one headline finding on a screen, flush in the content column
//
// `Card` keeps its old signature (default tier "panel"), so every existing
// call site renders as before.

const TIERS = {
  panel: "surface-panel",
  readout: "surface-readout",
};

export default function Card({ tier = "panel", children, className = "" }) {
  return <div className={`${TIERS[tier] ?? TIERS.panel} ${className}`}>{children}</div>;
}

// A panel's own header. The old code repeated `px-6 py-4 border-b
// border-line-soft` with a bold div and an optional note in eight files.
export function PanelHead({ title, note, right, className = "" }) {
  return (
    <div
      className={`flex items-start justify-between gap-4 px-6 py-4 border-b border-line-soft ${className}`}
    >
      <div className="min-w-0">
        <div className="font-semibold text-lead">{title}</div>
        {note && <p className="text-fine text-ink-mid mt-0.5">{note}</p>}
      </div>
      {right}
    </div>
  );
}

// The headline finding. Flush left edge with a 3px state rule, no left radius:
// it reads as part of the content column rather than as another card.
//
// tone: "agree" | "disagree" | "caution"
const TONES = {
  agree: { wrap: "bg-ok-bg border-ok/25", rule: "bg-ok", head: "text-ok" },
  disagree: { wrap: "bg-blush-soft border-line", rule: "bg-accent", head: "text-accent" },
  caution: { wrap: "bg-warm border-warm-ink/25", rule: "bg-warm-ink", head: "text-warm-ink" },
};

export function Verdict({ tone = "disagree", headline, children }) {
  const t = TONES[tone] ?? TONES.disagree;
  return (
    <div className={`flex overflow-hidden rounded-r-panel border border-l-0 ${t.wrap}`}>
      <div className={`w-[3px] shrink-0 ${t.rule}`} aria-hidden="true" />
      <div className="px-6 py-4 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className={`font-display text-sub font-semibold ${t.head}`}>{headline}</span>
        {children && <span className="text-base text-ink-mid">{children}</span>}
      </div>
    </div>
  );
}

// Reading of this photo's own values, in plain language. Replaces the
// "What this means" block, minus its tracked-out mono eyebrow -- the crimson
// rule and the tinted ground already say what it is.
export function Reading({ children }) {
  return (
    <div className="flex overflow-hidden rounded-r-panel border border-l-0 border-line bg-blush-soft">
      <div className="w-[3px] shrink-0 bg-accent/40" aria-hidden="true" />
      <div className="px-6 py-4 text-base space-y-2">{children}</div>
    </div>
  );
}
