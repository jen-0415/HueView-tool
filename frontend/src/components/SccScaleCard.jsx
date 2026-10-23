import Card, { PanelHead } from "./Card";
import { SCC, SCC_MEANING } from "../constants";

// What the predicted class actually is. A result that reads "SCC-3 Medium"
// means nothing on its own, so the scale is spelled out next to it: the six
// clusters, and the MST steps each one covers. The upload screen explains the
// same thing from the same constants, for anyone who reads it before uploading.
//
// The swatches are measured colour, so this is the one place on the summary
// where saturated non-palette colour is allowed to be loud.
export default function SccScaleCard({ predicted }) {
  return (
    <Card>
      <PanelHead title="What the SCC classes are" note={SCC_MEANING.short} />

      <div className="px-6 py-5">
        <p className="text-fine leading-relaxed text-ink-mid">{SCC_MEANING.basis}</p>

        <ul className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mt-5">
          {SCC.map((s) => {
            const on = s.id === predicted;
            return (
              <li
                key={s.id}
                className={`rounded-readout border px-3 py-3 transition-colors ${
                  on ? "border-accent bg-blush-soft shadow-readout" : "border-line-soft"
                }`}
              >
                <span
                  className="block w-full h-8 rounded border border-rule/60"
                  style={{ background: s.hex }}
                  role="img"
                  aria-label={`${s.id} colour`}
                />
                <span
                  className={`block font-mono text-data mt-2.5 ${
                    on ? "text-accent font-semibold" : "text-ink-mid"
                  }`}
                >
                  {s.id}
                  {on && <span className="sr-only"> — predicted for this photo</span>}
                </span>
                <span className="block text-fine leading-tight mt-0.5">{s.label}</span>
                <span className="block font-mono text-data text-ink-mid mt-1">{s.mst}</span>
              </li>
            );
          })}
        </ul>

        <p className="text-fine leading-relaxed text-ink-mid mt-5">
          Neighbouring MST steps share a cluster, so two faces one MST step apart can land in the
          same SCC class. The ground truth for every test photo is its MST label mapped this way
          (<span className="font-mono text-data">add_scc_labels.py</span>), not a separate SCC
          annotation.
        </p>
      </div>
    </Card>
  );
}
