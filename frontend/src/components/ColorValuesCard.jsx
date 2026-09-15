import Card from "./Card";
import RgbBars from "./RgbBars";
import LabBars from "./LabBars";

// Color values live here, not inside the model cards. Each pipeline shows the
// space it actually works in: the baseline averages RGB, HueView converts the
// selected regions to CIELAB.
export default function ColorValuesCard({ baseline, hueview }) {
    return (
        <Card>
            <div className="px-6 py-4 border-b border-line-soft">
                <div className="font-semibold">Color values</div>
                <div className="text-xs text-ink-soft">
                    Each pipeline reports the color space it operates in.
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-line-soft">
                <div className="px-6 py-5">
                    <div className="font-mono text-[10px] tracking-widest text-ink-soft mb-1">
                        BASELINE — RGB
                    </div>
                    <div className="text-[12px] text-ink-soft mb-4">Global RGB mean, full crop</div>
                    <RgbBars rgb={baseline.rgb} />
                </div>

                <div className="px-6 py-5">
                    <div className="font-mono text-[10px] tracking-widest text-accent mb-1">
                        HUEVIEW — CIELAB
                    </div>
                    <div className="text-[12px] text-ink-soft mb-4">
                        Mean of the selected regional values
                    </div>
                    <LabBars lab={hueview.lab} />
                </div>
            </div>
        </Card>
    );
}