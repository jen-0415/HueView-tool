import Card from "./Card";

// All six regions from SOP 3, with the Phase 7.5 selector's decision visible.
export default function RegionTable({ regions }) {
  return (
    <Card>
      <div className="px-6 py-4 border-b border-line-soft font-semibold">
        Color by face area (HueView only)
      </div>

      <table className="w-full font-mono text-xs">
        <thead>
          <tr className="bg-blush">
            {["Area", "L*", "a*", "b*", "Skin pixels", "Used"].map((h) => (
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
        &ldquo;Excluded&rdquo; areas had too little skin to measure, so values from the other
        areas fill in for them. The Full Face row sums up the five areas; its own model still
        counts toward the final result.
      </p>
    </Card>
  );
}
