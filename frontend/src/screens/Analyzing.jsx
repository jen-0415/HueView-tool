import { STAGES } from "../constants";

// Progress is driven by real stage events from the backend, not a timer.
export default function Analyzing({ stages }) {
  const doneCount = STAGES.filter((s) => stages[s.key]?.status === "done").length;
  const pct = Math.round((doneCount / STAGES.length) * 100);

  return (
    <main className="max-w-3xl mx-auto px-8 py-14">
      <div className="flex items-center gap-5">
        <Ring pct={pct} />
        <h2 className="font-display italic text-3xl">Analyzing photo</h2>
      </div>

      <ol className="flex flex-col gap-3 mt-10">
        {STAGES.map((s, i) => {
          const st = stages[s.key]?.status;
          const done = st === "done";
          const running = st === "running";

          return (
            <li
              key={s.key}
              className={`rounded-xl px-5 py-4 flex justify-between items-center border ${
                done ? "bg-blush-soft border-line-soft" : "bg-white"
              } ${running ? "border-accent" : "border-line-soft"}`}
            >
              <span className="flex items-center gap-4">
                <span
                  className={`w-7 h-7 rounded-full grid place-items-center font-mono text-xs ${
                    done ? "bg-accent text-white" : "border border-line text-accent"
                  }`}
                >
                  {done ? "✓" : i + 1}
                </span>
                <span className="text-[15px]">{s.label}</span>
              </span>

              <span
                className={`font-mono text-[11px] ${
                  done ? "text-ok" : running ? "text-accent" : "text-ink-soft"
                }`}
              >
                {done ? `${stages[s.key].ms} ms` : running ? "running…" : "queued"}
              </span>
            </li>
          );
        })}
      </ol>

      <p className="text-[13px] text-ink-soft mt-8">
        Stages 2–6 apply to HueView only. The baseline classifies the same crop directly.
      </p>
    </main>
  );
}

function Ring({ pct }) {
  const r = 26;
  const c = 2 * Math.PI * r;

  return (
    <div className="relative w-[68px] h-[68px]">
      <svg width="68" height="68">
        <circle cx="34" cy="34" r={r} fill="none" stroke="#F8DDE4" strokeWidth="6" />
        <circle
          cx="34" cy="34" r={r} fill="none" stroke="#C9145C" strokeWidth="6"
          strokeLinecap="round" strokeDasharray={c}
          strokeDashoffset={c * (1 - pct / 100)}
          transform="rotate(-90 34 34)"
          className="transition-all duration-500"
        />
      </svg>
      <span className="absolute inset-0 grid place-items-center font-mono text-xs text-accent">
        {pct}%
      </span>
    </div>
  );
}
