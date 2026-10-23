import AxisBar from "./AxisBar";

// Baseline colour output: the global RGB mean, 0–255 per channel.
const CHANNELS = [
  { key: "R", label: "R", axis: "Red channel", track: "linear-gradient(90deg,#1B1416,#C62828)" },
  { key: "G", label: "G", axis: "Green channel", track: "linear-gradient(90deg,#1B1416,#2E7D4F)" },
  { key: "B", label: "B", axis: "Blue channel", track: "linear-gradient(90deg,#1B1416,#2F5EA8)" },
];

export default function RgbBars({ rgb, arriving = false }) {
  return (
    <div className="flex flex-col gap-5">
      {CHANNELS.map((c, i) => (
        <AxisBar
          key={c.key}
          label={c.label}
          axis={c.axis}
          value={rgb[c.key]}
          min={0}
          max={255}
          track={c.track}
          fill
          arriving={arriving}
          index={i}
        />
      ))}
    </div>
  );
}
