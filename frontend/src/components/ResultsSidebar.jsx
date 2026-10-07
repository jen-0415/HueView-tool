import Card from "./Card";
import { STEPS } from "../stages";

// Stage navigation for the Results screen. Every stage stays reachable after
// the analysis, so the user can go back and see what each step did.
// Below lg it collapses to a single <select> so it doesn't push the content
// a full screen down on a phone.
export default function ResultsSidebar({ stage, onPick, preview, onReset }) {
  const stepIdx = STEPS.findIndex((s) => s.id === stage);

  return (
    <nav
      aria-label="Result stages"
      className="w-full lg:w-[270px] lg:shrink-0 flex flex-col gap-4 lg:sticky lg:top-6"
    >
      <Card className="p-3.5 flex gap-3 items-center">
        {preview && (
          <img src={preview} alt="" className="w-16 h-16 rounded-lg object-cover shrink-0" />
        )}
        <button
          onClick={onReset}
          className="rounded-full bg-white border border-line text-accent text-xs px-3 min-h-[36px]"
        >
          + New analysis
        </button>
      </Card>

      {/* Phone / tablet: one dropdown + progress ticks. */}
      <div className="lg:hidden flex flex-col gap-2">
        <label className="font-mono text-[11px] text-ink-soft" htmlFor="stage-select">
          {stepIdx >= 0 ? `STAGE ${stepIdx + 1} OF ${STEPS.length}` : "STAGE"}
        </label>
        <select
          id="stage-select"
          value={stage}
          onChange={(e) => onPick(e.target.value)}
          className="min-h-[48px] rounded-xl border border-line bg-white px-3 text-[15px] font-semibold"
        >
          <option value="summary">Summary</option>
          <optgroup label="HueView pipeline">
            {STEPS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.n} {s.label}
              </option>
            ))}
          </optgroup>
          <optgroup label="Compare">
            <option value="compare">Baseline vs HueView</option>
          </optgroup>
        </select>
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${STEPS.length}, minmax(0,1fr))` }}>
          {STEPS.map((s, i) => (
            <div key={s.id} className={`h-1 rounded-sm ${i <= stepIdx ? "bg-accent" : "bg-line"}`} />
          ))}
        </div>
      </div>

      {/* Desktop: full list. */}
      <Card className="hidden lg:flex p-2.5 flex-col gap-0.5">
        <NavButton active={stage === "summary"} onClick={() => onPick("summary")} mark="★">
          Summary
        </NavButton>

        <GroupLabel>HueView pipeline</GroupLabel>
        {STEPS.map((s) => (
          <NavButton key={s.id} active={stage === s.id} onClick={() => onPick(s.id)} mark={s.n}>
            {s.label}
          </NavButton>
        ))}

        <GroupLabel>Compare</GroupLabel>
        <NavButton active={stage === "compare"} onClick={() => onPick("compare")} mark="⇄">
          Baseline vs HueView
        </NavButton>
      </Card>

      <p className="hidden lg:block text-xs text-ink-soft leading-relaxed px-1">
        Every stage stays here after the analysis, so you can go back and see what each step did
        to your photo.
      </p>
    </nav>
  );
}

function GroupLabel({ children }) {
  return (
    <div className="font-mono text-[11px] tracking-widest uppercase text-ink-soft px-3 pt-3.5 pb-1.5">
      {children}
    </div>
  );
}

function NavButton({ active, onClick, mark, children }) {
  return (
    <button
      onClick={onClick}
      aria-current={active ? "step" : undefined}
      className={`flex items-center gap-2.5 w-full min-h-[44px] px-3 py-2 rounded-[10px] text-left text-sm ${
        active ? "bg-accent text-white" : "text-ink hover:bg-blush-soft"
      }`}
    >
      <span className={`font-mono text-[11px] w-5 ${active ? "text-[#FFE3EE]" : "text-ink-soft"}`}>
        {mark}
      </span>
      <span className="flex-1">{children}</span>
    </button>
  );
}
