import Card from "./Card";
import { SCC, sccLabel } from "../constants";

// HueView's regional path, end to end: where each region is on the face,
// which pixels survived the skin filter, what each of the six models saw and
// predicted, and how those six predictions become the final SCC.
export default function RegionalSegmentationCard({ hueview }) {
  const seg = hueview.segmentation;
  const regions = hueview.regions ?? [];
  const hasCls = regions.some((r) => Array.isArray(r.probabilities));
  const colorOf = (key) => seg?.colors?.[key] ?? "#C9145C";

  return (
    <Card>
      <div className="px-6 py-4 border-b border-line-soft">
        <div className="font-semibold">Regional segmentation &amp; per-region SCC — HueView</div>
        <div className="text-xs text-ink-soft">
          MediaPipe landmarks → convex-hull regions → HSV skin filter → one model per region.
        </div>
      </div>

      {seg && (
        <div className="px-6 py-5 grid grid-cols-1 sm:grid-cols-[auto_auto_1fr] gap-6 items-start">
          <Figure n="01" src={seg.geometric} caption="Geometric regions (landmark convex hulls)" />
          <Figure n="02" src={seg.skin} caption="Skin kept by the HSV filter (removed pixels darkened)" />
          <ul className="text-[13px] space-y-2 self-center">
            {regions
              .filter((r) => seg.colors[r.region_key])
              .map((r) => (
                <li key={r.region_key} className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-sm" style={{ background: colorOf(r.region_key) }} />
                  <span>{r.name}</span>
                  <span className="font-mono text-[11px] text-ink-soft">
                    {r.pixels.toLocaleString()} / {seg.geometric_pixels[r.region_key].toLocaleString()} px skin
                  </span>
                </li>
              ))}
          </ul>
        </div>
      )}

      <div className="px-6 pb-5">
        <div className="font-mono text-[11px] font-semibold mb-3">03 · What each model saw, and its SCC</div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {regions.map((r) => (
            <div key={r.region_key ?? r.name} className="rounded-xl border border-line-soft p-2">
              {seg?.inputs?.[r.region_key] && (
                <img
                  src={seg.inputs[r.region_key]}
                  alt={`${r.name} model input`}
                  className="w-full aspect-square object-cover rounded-lg border-2"
                  style={{ borderColor: colorOf(r.region_key) }}
                />
              )}
              <div className="text-[12px] mt-2 leading-tight">{r.name}</div>
              <div className="flex items-center gap-2 mt-1">
                {r.scc && (
                  <span
                    className="w-4 h-4 rounded border border-line-soft"
                    style={{ background: SCC.find((s) => s.id === r.scc)?.hex }}
                  />
                )}
                <span className="font-mono text-[12px] font-semibold">{r.scc ?? "—"}</span>
                <span className="font-mono text-[11px] text-ink-soft">
                  {r.confidence != null ? `${Math.round(r.confidence * 100)}%` : ""}
                </span>
              </div>
              {r.status && r.status !== "ok" && (
                <div className="font-mono text-[10px] text-warm-ink mt-1">status: {r.status}</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {hasCls && <DecisionTable hueview={hueview} regions={regions} />}
    </Card>
  );
}

function Figure({ n, src, caption }) {
  return (
    <figure className="w-[224px]">
      <img src={src} alt={caption} className="w-[224px] h-[224px] rounded-lg" />
      <figcaption className="mt-2">
        <div className="font-mono text-[11px] font-semibold">{n}</div>
        <div className="text-[12px] text-ink-soft leading-snug">{caption}</div>
      </figcaption>
    </figure>
  );
}

// Every model's probability vector, then their mean -- the final SCC is the
// largest mean. Shown in full so the decision can be checked by eye.
function DecisionTable({ hueview, regions }) {
  const final = hueview.probabilities;
  const pct = (p) => (p == null ? "—" : `${(p * 100).toFixed(1)}%`);
  const argmax = (ps) => ps.indexOf(Math.max(...ps));

  return (
    <div className="border-t border-line-soft">
      <div className="px-6 pt-4">
        <div className="font-mono text-[11px] font-semibold">04 · How the final SCC is decided</div>
        <p className="text-[12px] text-ink-soft mt-1 max-w-3xl">{hueview.decision?.description}</p>
      </div>
      <div className="overflow-x-auto px-6 py-4">
        <table className="w-full font-mono text-xs">
          <thead>
            <tr className="bg-blush">
              <th className="px-3 py-2 text-left font-normal text-[10px] tracking-widest text-accent">Model</th>
              {SCC.map((s) => (
                <th key={s.id} className="px-3 py-2 text-right font-normal text-[10px] tracking-widest text-accent">
                  {s.id}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {regions.map((r) => {
              const top = argmax(r.probabilities);
              return (
                <tr key={r.region_key ?? r.name} className="border-t border-line-soft">
                  <td className="px-3 py-2">{r.name}</td>
                  {r.probabilities.map((p, i) => (
                    <td key={i} className={`px-3 py-2 text-right ${i === top ? "font-semibold" : "text-ink-soft"}`}>
                      {pct(p)}
                    </td>
                  ))}
                </tr>
              );
            })}
            {final && (
              <tr className="border-t-2 border-accent bg-blush-soft">
                <td className="px-3 py-2 font-semibold text-accent">Mean of 6 → final</td>
                {final.map((p, i) => (
                  <td
                    key={i}
                    className={`px-3 py-2 text-right ${i === argmax(final) ? "font-semibold text-accent" : ""}`}
                  >
                    {pct(p)}
                  </td>
                ))}
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {hueview.scc && (
        <p className="px-6 pb-5 text-[13px]">
          Final HueView SCC: <span className="font-semibold text-accent">{hueview.scc} — {sccLabel(hueview.scc)}</span>
          <span className="text-ink-soft"> (highest mean probability, {pct(hueview.confidence)})</span>
        </p>
      )}
    </div>
  );
}
