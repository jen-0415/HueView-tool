import { measureCls, stagger } from "../motion";

// One labelled axis with a marker at the measured value. RgbBars and LabBars
// were the same component twice over with different ranges, so they share this
// one now; each still names its own channels and real ranges.
//
// `fill` draws a rail from the axis origin to the marker. RGB channels run
// 0..255 from one end, so a rail reads correctly; a* and b* are signed around
// a midpoint, where a rail from the left end would imply the wrong thing, so
// those pass fill={false} and animate the marker alone.
export default function AxisBar({
  label,
  axis,
  value,
  min,
  max,
  track,
  fill = false,
  arriving = false,
  index = 0,
}) {
  const known = typeof value === "number" && Number.isFinite(value);
  const pct = known ? ((value - min) / (max - min)) * 100 : null;

  return (
    <div>
      <div className="flex justify-between items-baseline gap-3">
        <span className="font-mono text-fine text-accent">{label}</span>
        <span className="font-mono text-lead">{known ? value : "—"}</span>
      </div>
      <div className="text-data text-ink-mid">{axis}</div>

      <div
        className="relative h-2.5 rounded-full mt-2"
        style={{ background: track }}
        role="img"
        aria-label={`${label} ${known ? value : "not available"} on a ${min} to ${max} scale`}
      >
        {pct !== null && fill && (
          <span
            className={`absolute inset-y-0 left-0 rounded-full bg-ink/15 ${measureCls(arriving)}`}
            style={{ width: `${pct}%`, transformOrigin: "left center", ...stagger(index) }}
          />
        )}
        {pct !== null && (
          <span
            className={`absolute top-1/2 w-1.5 h-5 rounded-full bg-ink border border-white shadow-sm ${
              arriving ? "cue-in" : ""
            }`}
            style={{ left: `${pct}%`, transform: "translate(-50%,-50%)", ...stagger(index) }}
          />
        )}
      </div>
      <div className="flex justify-between font-mono text-data text-ink-mid mt-1.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
