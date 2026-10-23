import AxisBar from "./AxisBar";

// CIELAB values as labelled axis bars. Ranges are the real CIELAB ranges:
// L* 0..100, a*/b* -128..127. a* and b* are signed around a midpoint, so they
// show a marker without a rail.
const AXES = [
  {
    key: "L", label: "L*", axis: "Lightness", min: 0, max: 100, fill: true,
    track: "linear-gradient(90deg,#1B1416,#FFFFFF)",
  },
  {
    key: "a", label: "a*", axis: "Green to red axis", min: -128, max: 127, fill: false,
    track: "linear-gradient(90deg,#2E7D4F,#EFEFEF,#C62828)",
  },
  {
    key: "b", label: "b*", axis: "Blue to yellow axis", min: -128, max: 127, fill: false,
    track: "linear-gradient(90deg,#2F5EA8,#EFEFEF,#D9A400)",
  },
];

export default function LabBars({ lab, arriving = false }) {
  return (
    <div className="flex flex-col gap-5">
      {AXES.map((a, i) => (
        <AxisBar
          key={a.key}
          label={a.label}
          axis={a.axis}
          // null when no region was usable; AxisBar renders an em dash.
          value={lab?.[a.key]}
          min={a.min}
          max={a.max}
          track={a.track}
          fill={a.fill}
          arriving={arriving}
          index={i}
        />
      ))}
    </div>
  );
}
