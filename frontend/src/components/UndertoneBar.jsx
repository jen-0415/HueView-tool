const PARTS = [
  { key: "warm", label: "Warm", bg: "bg-warm", swatch: "#F7EDC0" },
  { key: "neutral", label: "Neutral", bg: "bg-neutraltone", swatch: "#F1E4EA" },
  { key: "cool", label: "Cool", bg: "bg-cool", swatch: "#C6D6DD" },
];

// Labels live in the legend, never inside the segments — a 7% segment
// cannot hold text, and clipped words look like a rendering bug.
export default function UndertoneBar({ distribution }) {
  // null when no region had usable skin pixels -- nothing to plot.
  if (!distribution) {
    return <div className="font-mono text-[11px] text-ink-soft">No usable regions</div>;
  }
  return (
    <div>
      <div className="flex h-3 rounded-full overflow-hidden">
        {PARTS.map((p) => (
          <div
            key={p.key}
            className={p.bg}
            style={{ width: `${distribution[p.key] * 100}%` }}
          />
        ))}
      </div>

      <ul className="flex flex-wrap gap-x-4 gap-y-1 mt-3">
        {PARTS.map((p) => (
          <li key={p.key} className="flex items-center gap-2 font-mono text-[11px]">
            <span
              className="w-3 h-3 rounded-sm border border-line-soft"
              style={{ background: p.swatch }}
            />
            <span className="text-ink-soft">{p.label}</span>
            <span>{Math.round(distribution[p.key] * 100)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}