// CIELAB values as labelled axis bars, matching the original mockup.
// Ranges are the real CIELAB ranges: L* 0..100, a*/b* -128..127.
const AXES = [
    {
        key: "L", label: "L*", axis: "Lightness", min: 0, max: 100,
        track: "linear-gradient(90deg,#1B1416,#FFFFFF)"
    },
    {
        key: "a", label: "a*", axis: "Green to Red axis", min: -128, max: 127,
        track: "linear-gradient(90deg,#2E7D4F,#EFEFEF,#C62828)"
    },
    {
        key: "b", label: "b*", axis: "Blue to Yellow axis", min: -128, max: 127,
        track: "linear-gradient(90deg,#2F5EA8,#EFEFEF,#D9A400)"
    },
];

export default function LabBars({ lab }) {
    return (
        <div className="flex flex-col gap-4">
            {AXES.map((a) => {
                const value = lab[a.key];
                const pct = ((value - a.min) / (a.max - a.min)) * 100;

                return (
                    <div key={a.key}>
                        <div className="flex justify-between items-baseline">
                            <span className="font-mono text-[12px] text-accent">{a.label}</span>
                            <span className="font-mono text-[15px]">{value}</span>
                        </div>
                        <div className="text-[11px] text-ink-soft">{a.axis}</div>

                        <div
                            className="relative h-2 rounded-full mt-1.5"
                            style={{ background: a.track }}
                            role="img"
                            aria-label={`${a.label} ${value}`}
                        >
                            <span
                                className="absolute top-1/2 w-1 h-4 rounded-full bg-ink border border-white"
                                style={{ left: `${pct}%`, transform: "translate(-50%,-50%)" }}
                            />
                        </div>
                    </div>
                );
            })}
        </div>
    );
}