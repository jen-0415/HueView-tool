import Card from "./Card";

// Phase 3 k-means illumination bin for the uploaded image.
// This is a property of the input, not of either pipeline, so it sits with
// the photo rather than inside a model card.
export default function IlluminationCard({ illumination }) {
    const { value, scale, bin, bins, metric } = illumination;
    const [min, max] = scale;
    const pos = ((value - min) / (max - min)) * 100;

    return (
        <Card>
            <div className="px-5 py-4">
                <div className="font-mono text-[10px] tracking-widest text-ink-soft">
                    ILLUMINATION BIN
                </div>

                <div className="flex items-baseline gap-2 mt-1">
                    <span className="font-display text-[22px] text-accent">{bin}</span>
                    <span className="font-mono text-[11px] text-ink-soft">{value}</span>
                </div>

                <div className="text-[11px] text-ink-soft">{metric}</div>

                {/* The three bins drawn to their real widths, with the image's position. */}
                <div className="relative mt-3">
                    <div className="flex h-2.5 rounded-full overflow-hidden">
                        {bins.map((b) => {
                            const width = ((b.range[1] - b.range[0]) / (max - min)) * 100;
                            const active = b.name === bin;
                            return (
                                <div
                                    key={b.name}
                                    style={{
                                        width: `${width}%`,
                                        background: active ? "#C9145C" : "#F8DDE4",
                                    }}
                                />
                            );
                        })}
                    </div>

                    <span
                        className="absolute top-1/2 w-1 h-4 rounded-full bg-ink border border-white"
                        style={{ left: `${pos}%`, transform: "translate(-50%,-50%)" }}
                    />
                </div>

                <ul className="flex justify-between mt-2">
                    {bins.map((b) => (
                        <li
                            key={b.name}
                            className={`font-mono text-[9px] ${b.name === bin ? "text-accent" : "text-ink-soft"
                                }`}
                        >
                            {b.name}
                        </li>
                    ))}
                </ul>

                <p className="text-[11px] text-ink-soft leading-snug mt-3">
                    Bin {illumination.bin_index} of {bins.length}, assigned by k-means on mean
                    luminance. Used to stratify evaluation, not to alter the prediction.
                </p>
            </div>
        </Card>
    );
}