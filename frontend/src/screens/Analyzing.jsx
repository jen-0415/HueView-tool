import { ANALYSIS_STAGES } from "../stages";

// Loading screen while the backend analyzes the photo.
//
// The backend already streams a "stage" event per step and the frontend was
// throwing them away, so this showed a bare spinner that said nothing. It now
// names the step that is running.
//
// Only the step's state is shown, never a duration: routes.py runs detect and
// ssr for real and treats the remaining four as progress beats, so their
// per-stage timings are not individually measured. The real per-model cost is
// in the result payload.
//
// The SSR intermediate images are not shown here; they appear in the Results
// screen's lighting-correction stage.
export default function Analyzing({ stages = {} }) {
  const done = ANALYSIS_STAGES.filter((s) => stages[s.key] === "done").length;
  const running = ANALYSIS_STAGES.find((s) => stages[s.key] === "running");

  return (
    <main className="min-h-[70vh] grid place-items-center px-4 sm:px-8 py-12">
      <div className="w-full max-w-sm">
        <h2 className="font-display text-head font-semibold">
          {running ? running.label : "Analysing photo"}
        </h2>
        <p className="text-base text-ink-mid mt-2" role="status" aria-live="polite">
          {done === ANALYSIS_STAGES.length
            ? "Building the result…"
            : `Step ${Math.min(done + 1, ANALYSIS_STAGES.length)} of ${ANALYSIS_STAGES.length}`}
        </p>

        <ol className="flex flex-col mt-7">
          {ANALYSIS_STAGES.map((s) => {
            const state = stages[s.key] ?? "waiting";
            return (
              <li
                key={s.key}
                className={`flex items-center gap-3.5 py-2.5 text-fine transition-colors duration-200 ${
                  state === "waiting" ? "text-ink-mid" : "text-ink"
                }`}
              >
                <Marker state={state} />
                <span className={state === "running" ? "font-semibold" : ""}>{s.label}</span>
              </li>
            );
          })}
        </ol>

        <div className="h-1 rounded-full bg-line mt-5 overflow-hidden">
          <div
            className="h-full bg-accent rounded-full transition-[width] duration-500 ease-settle"
            style={{ width: `${(done / ANALYSIS_STAGES.length) * 100}%` }}
          />
        </div>
      </div>
    </main>
  );
}

function Marker({ state }) {
  if (state === "done") {
    return (
      <span
        className="w-5 h-5 shrink-0 rounded-full bg-accent grid place-items-center"
        aria-hidden="true"
      >
        <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="#fff" strokeWidth="2">
          <path d="M2.5 6.5 5 9l4.5-5.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
    );
  }
  if (state === "running") {
    return (
      <span
        className="w-5 h-5 shrink-0 rounded-full border-2 border-line border-t-accent animate-spin"
        aria-hidden="true"
      />
    );
  }
  return (
    <span className="w-5 h-5 shrink-0 rounded-full border-2 border-line-soft" aria-hidden="true" />
  );
}
