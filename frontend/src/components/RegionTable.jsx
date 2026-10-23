import Card, { PanelHead } from "./Card";

// All six regions from SOP 3, with the Phase 7.5 selector's decision visible.
export default function RegionTable({ regions }) {
  return (
    <Card>
      <PanelHead
        title="Colour by face area"
        note="ChroLumi only. Areas with too little skin to measure are dimmed."
      />

      {/* The table had no scroll container, so on a phone it pushed out of the
          panel. Headers stay put while the rows scroll under them. */}
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-data min-w-[520px]">
          <thead>
            <tr className="bg-blush">
              {["Area", "Lightness (L*)", "Redness (a*)", "Yellowness (b*)", "Skin pixels"].map(
                (h, i) => (
                  <th
                    key={h}
                    className={`sticky top-0 bg-blush px-5 py-3 font-medium text-fine text-accent ${
                      i === 0 ? "text-left" : "text-right"
                    }`}
                  >
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {regions.map((r) => {
              // Full Face is the five areas' average, so the backend marks it
              // used: false; it is not dimmed.
              const whole = r.region_key === "full_face";
              return (
                <tr
                  key={r.name}
                  className={`border-t border-rule/60 ${
                    r.used || whole ? "" : "opacity-55"
                  } ${whole ? "bg-blush-soft/60" : ""}`}
                >
                  <td className="px-5 py-3 text-ink font-sans text-fine">{r.name}</td>
                  <td className="px-5 py-3 text-right">{r.L ?? "—"}</td>
                  <td className="px-5 py-3 text-right">{r.a ?? "—"}</td>
                  <td className="px-5 py-3 text-right">{r.b ?? "—"}</td>
                  <td className="px-5 py-3 text-right text-ink-mid">
                    {r.pixels.toLocaleString()}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
