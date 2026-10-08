import Card from "./Card";
import { SCC, SCC_MEANING } from "../constants";

// What the predicted class actually is. A result that reads "SCC-3 — Medium"
// means nothing on its own, so the scale is spelled out next to it: the six
// clusters, and the MST steps each one covers. The upload screen explains the
// same thing from the same constants, for anyone who reads it before uploading.
export default function SccScaleCard({ predicted }) {
  return (
    <Card>
      <div className="px-6 py-4 border-b border-line-soft">
        <div className="font-semibold">What the SCC classes are</div>
        <div className="text-xs text-ink-soft">{SCC_MEANING.short}</div>
      </div>

      <div className="px-6 py-5">
        <p className="text-[13px] leading-relaxed text-ink-soft max-w-3xl">
          {SCC_MEANING.basis}
        </p>

        <ul className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mt-4">
          {SCC.map((s) => {
            const on = s.id === predicted;
            return (
              <li
                key={s.id}
                className={`rounded-xl border px-3 py-2.5 ${
                  on ? "border-accent bg-blush-soft" : "border-line-soft"
                }`}
              >
                <span
                  className="block w-full h-5 rounded"
                  style={{ background: s.hex }}
                  role="img"
                  aria-label={`${s.id} colour`}
                />
                <span className={`block font-mono text-[11px] mt-2 ${on ? "text-accent font-semibold" : "text-ink-soft"}`}>
                  {s.id}
                </span>
                <span className="block text-[13px] leading-tight">{s.label}</span>
                <span className="block font-mono text-[10px] text-ink-soft mt-1">{s.mst}</span>
              </li>
            );
          })}
        </ul>

        <p className="text-[12px] text-ink-soft mt-4">
          Neighbouring MST steps share a cluster, so two faces one MST step apart can land in the
          same SCC class. The ground truth for every test photo is its MST label mapped this way
          (<span className="font-mono">add_scc_labels.py</span>), not a separate SCC annotation.
        </p>
      </div>
    </Card>
  );
}
