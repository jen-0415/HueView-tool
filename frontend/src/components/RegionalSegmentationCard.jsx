import Card from "./Card";
import { SCC, sccLabel } from "../constants";

// HueView's regional path, end to end: where each region is on the face,
// which pixels survived the skin filter, what each of the six models saw and
// predicted, and how their votes become the final SCC.
export default function RegionalSegmentationCard({ hueview }) {
  return (
    <Card>
      <div className="px-6 py-4 border-b border-line-soft">
        <div className="font-semibold">Regional segmentation &amp; per-region SCC — HueView</div>
        <div className="text-xs text-ink-soft">
          MediaPipe landmarks → convex-hull regions → HSV skin filter → one model per region.
        </div>
      </div>
      <SegmentationFigures hueview={hueview} />
      <RegionPredictions hueview={hueview} />
      <DecisionTable hueview={hueview} />
    </Card>
  );
}

// The pieces below are exported separately so the Results stage panels can
// show one step at a time. Each renders nothing when its data is missing.

const colorFor = (seg) => (key) => seg?.colors?.[key] ?? "#C9145C";

export function SegmentationFigures({ hueview }) {
  const seg = hueview.segmentation;
  const regions = hueview.regions ?? [];
  const colorOf = colorFor(seg);
  if (!seg) return null;

  return (
    <div className="px-6 py-5 grid grid-cols-1 sm:grid-cols-[auto_auto_1fr] gap-6 items-start">
      <Figure n="01" src={seg.geometric} caption="Face areas outlined from the mapped points" />
      <Figure n="02" src={seg.skin} caption="Skin that was kept (removed parts darkened)" />
      <ul className="text-[13px] space-y-2 self-center">
        {regions
          .filter((r) => seg.colors[r.region_key])
          .map((r) => (
            <li key={r.region_key} className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-sm" style={{ background: colorOf(r.region_key) }} />
              <span>{r.name}</span>
              <span className="font-mono text-[11px] text-ink-soft">
                {r.pixels.toLocaleString()} / {seg.geometric_pixels[r.region_key].toLocaleString()} skin pixels
              </span>
            </li>
          ))}
      </ul>
    </div>
  );
}

export function RegionPredictions({ hueview }) {
  const seg = hueview.segmentation;
  const regions = hueview.regions ?? [];
  const colorOf = colorFor(seg);

  return (
      <div className="px-6 py-5">
        <div className="font-mono text-[11px] font-semibold mb-3">What each model saw, and what it picked</div>
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

// Every model's probability vector and vote, the vote tally, and the mean
// probability used only to break a tie. Shown in full so the decision can be
// checked by eye.
export function DecisionTable({ hueview }) {
  const regions = (hueview.regions ?? []).filter((r) => Array.isArray(r.probabilities));
  if (regions.length === 0) return null;
  const mean = hueview.probabilities;
  const votes = hueview.votes ?? {};
  const pct = (p) => (p == null ? "—" : `${(p * 100).toFixed(1)}%`);
  const argmax = (ps) => ps.indexOf(Math.max(...ps));
  const finalIdx = SCC.findIndex((s) => s.id === hueview.scc);
  const headCls = "px-3 py-2 text-right font-normal text-[10px] tracking-widest text-accent";

  return (
    <div className="border-t border-line-soft">
      <div className="px-6 pt-4">
        <div className="font-mono text-[11px] font-semibold">How the final answer is chosen</div>
        <p className="text-[12px] leading-relaxed text-ink-soft mt-1 text-justify">{hueview.decision?.description}</p>
      </div>
      <div className="overflow-x-auto px-6 py-4">
        <table className="w-full font-mono text-xs">
          <thead>
            <tr className="bg-blush">
              <th className="px-3 py-2 text-left font-normal text-[10px] tracking-widest text-accent">Model</th>
              {SCC.map((s) => (
                <th key={s.id} className={headCls}>
                  {s.id}
                </th>
              ))}
              <th className={headCls}>Vote</th>
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
                  <td className="px-3 py-2 text-right font-semibold">{r.scc}</td>
                </tr>
              );
            })}
            {mean && (
              <tr className="border-t-2 border-accent">
                <td className="px-3 py-2 text-ink-soft">Average of {regions.length} (only used to break a tie)</td>
                {mean.map((p, i) => (
                  <td key={i} className="px-3 py-2 text-right text-ink-soft">
                    {pct(p)}
                  </td>
                ))}
                <td />
              </tr>
            )}
            {hueview.votes && (
              <tr className="border-t border-line-soft bg-blush-soft">
                <td className="px-3 py-2 font-semibold text-accent">Votes → final</td>
                {SCC.map((s, i) => (
                  <td
                    key={s.id}
                    className={`px-3 py-2 text-right ${i === finalIdx ? "font-semibold text-accent" : ""}`}
                  >
                    {votes[s.id] ?? 0}
                  </td>
                ))}
                <td className="px-3 py-2 text-right font-semibold text-accent">{hueview.scc}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {hueview.scc && (
        <p className="px-6 pb-5 text-[13px]">
          HueView&apos;s final answer: <span className="font-semibold text-accent">{hueview.scc} — {sccLabel(hueview.scc)}</span>
          <span className="text-ink-soft">
            {" "}
            ({votes[hueview.scc] ?? "?"} of {regions.length} models picked it
            {hueview.tied ? "; it was a tie, so the higher average score decided" : ""})
          </span>
        </p>
      )}
    </div>
  );
}
