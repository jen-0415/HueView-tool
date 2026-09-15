import Card from "./Card";

// All six regions from SOP 3, with the Phase 7.5 selector's decision visible.
export default function RegionTable({ regions }) {
  return (
    <Card>
      <div className="px-6 py-4 border-b border-line-soft font-semibold">
        Regional facial analysis — HueView only
      </div>

      <table className="w-full font-mono text-xs">
        <thead>
          <tr className="bg-blush">
            {["Region", "L*", "a*", "b*", "Pixels", "Used"].map((h) => (
              <th
                key={h}
                className="px-5 py-3 text-left font-normal tracking-widest text-[10px] text-accent"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {regions.map((r) => (
            <tr
              key={r.name}
              className={`border-t border-line-soft ${r.used ? "" : "opacity-50"}`}
            >
              <td className="px-5 py-3 text-ink">{r.name}</td>
              <td className="px-5 py-3">{r.L ?? "—"}</td>
              <td className="px-5 py-3">{r.a ?? "—"}</td>
              <td className="px-5 py-3">{r.b ?? "—"}</td>
              <td className="px-5 py-3">{r.pixels.toLocaleString()}</td>
              <td className={`px-5 py-3 ${r.used ? "text-ok" : "text-ink-soft"}`}>
                {r.used ? "yes" : "excluded"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="px-6 py-3 text-xs text-ink-soft">
        Selection by the Regional Configuration Selector. Full face is excluded when the
        five sub-regions pass the skin-pixel threshold.
      </p>
    </Card>
  );
}
