// Baseline color output: the global RGB mean, 0–255 per channel.
const CHANNELS = [
    { key: "R", label: "R", axis: "Red channel", track: "linear-gradient(90deg,#1B1416,#C62828)" },
    { key: "G", label: "G", axis: "Green channel", track: "linear-gradient(90deg,#1B1416,#2E7D4F)" },
    { key: "B", label: "B", axis: "Blue channel", track: "linear-gradient(90deg,#1B1416,#2F5EA8)" },
];

export default function RgbBars({ rgb }) {
    return (
        <div>
            <div className="flex flex-col gap-4">
                {CHANNELS.map((c) => {
                    const value = rgb[c.key];
                    return (
                        <div key={c.key}>
                            <div className="flex justify-between items-baseline">
                                <span className="font-mono text-[12px] text-accent">{c.label}</span>
                                <span className="font-mono text-[15px]">{value}</span>
                            </div>
                            <div className="text-[11px] text-ink-soft">{c.axis}</div>

                            <div
                                className="relative h-2 rounded-full mt-1.5"
                                style={{ background: c.track }}
                                role="img"
                                aria-label={`${c.label} ${value}`}
                            >
                                <span
                                    className="absolute top-1/2 w-1 h-4 rounded-full bg-ink border border-white"
                                    style={{ left: `${(value / 255) * 100}%`, transform: "translate(-50%,-50%)" }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}