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
            {["Area", "L*", "a*", "b*", "Skin pixels"].map((h) => (
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
          {regions.map((r) => {
            // Areas with too little skin are dimmed. Full Face is the five areas'
            // average, so the backend marks it used: false; it is not dimmed.
            const whole = r.region_key === "full_face";
            return (
            <tr
              key={r.name}
              className={`border-t border-line-soft ${r.used || whole ? "" : "opacity-50"}`}
            >
              <td className="px-5 py-3 text-ink">{r.name}</td>
              <td className="px-5 py-3">{r.L ?? "—"}</td>
              <td className="px-5 py-3">{r.a ?? "—"}</td>
              <td className="px-5 py-3">{r.b ?? "—"}</td>
              <td className="px-5 py-3">{r.pixels.toLocaleString()}</td>
            </tr>
            );
          })}
        </tbody>
      </table>

    </Card>
  );
}
