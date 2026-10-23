import Card from "./Card";
import { measureCls, stagger } from "../motion";

// Phase 3 k-means illumination bin for the uploaded image.
// This is a property of the input, not of either pipeline, so it sits with
// the photo rather than inside a model card. Readout tier: it is a measured
// value you are meant to read off.
export default function IlluminationCard({ illumination, arriving = false }) {
  const { value, scale, bin, bins, metric } = illumination;
  const [min, max] = scale;

  return (
    <Card tier="readout">
      <div className="px-5 py-5">
        <div className="text-data text-ink-mid">Lighting level</div>

        <div className="flex items-baseline gap-3 mt-1">
          <span className="font-display text-title font-semibold text-accent">{bin}</span>
          <span className="font-mono text-lead">{value}</span>
        </div>

        <p className="text-fine text-ink-mid mt-0.5">{metric}</p>

        {/* The three bins, to scale, with this photo's bin filled. */}
        <div className="mt-4">
          <div
            className={`flex h-3 rounded-full overflow-hidden border border-rule/50 ${measureCls(
              arriving,
            )}`}
            style={stagger(0)}
          >
            {bins.map((b) => (
              <div
                key={b.name}
                style={{
                  width: `${((b.range[1] - b.range[0]) / (max - min)) * 100}%`,
                  background: b.name === bin ? "#C9145C" : "#F8DDE4",
                }}
              />
            ))}
          </div>

          <ul className="flex justify-between mt-2">
            {bins.map((b) => (
              <li
                key={b.name}
                className={`font-mono text-data ${
                  b.name === bin ? "text-accent font-semibold" : "text-ink-mid"
                }`}
              >
                {b.name}
              </li>
            ))}
          </ul>
        </div>

        <p className="text-fine text-ink-mid leading-relaxed mt-4">
          Group {illumination.bin_index} of {bins.length}, based on the photo&apos;s average
          brightness. Used to compare results across lighting conditions. It does not change the
          result.
        </p>
      </div>
    </Card>
  );
}
