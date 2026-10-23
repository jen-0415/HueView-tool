import Card, { PanelHead } from "./Card";
import RgbBars from "./RgbBars";
import LabBars from "./LabBars";

// Colour values live here, not inside the model cards. Each pipeline shows the
// space it actually works in: the baseline averages RGB, ChroLumi converts the
// selected regions to CIELAB.
export default function ColorValuesCard({ baseline, hueview, arriving = false }) {
  return (
    <Card>
      <PanelHead
        title="Colour values"
        note="Each pipeline shows colour in the format it works with."
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-line-soft">
        <div className="px-6 py-5">
          <div className="font-semibold text-fine">Baseline &mdash; RGB</div>
          <p className="text-fine text-ink-mid mt-0.5 mb-5">
            Average colour of the whole cropped photo
          </p>
          <RgbBars rgb={baseline.rgb} arriving={arriving} />
        </div>

        <div className="px-6 py-5">
          <div className="font-semibold text-fine text-accent">ChroLumi &mdash; CIELAB</div>
          <p className="text-fine text-ink-mid mt-0.5 mb-5">
            Average across the face areas used
          </p>
          <LabBars lab={hueview.lab} arriving={arriving} />
        </div>
      </div>
    </Card>
  );
}
