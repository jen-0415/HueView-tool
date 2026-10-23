import Card from "./Card";
import { STEPS } from "../stages";

// Stage navigation for the Results screen. Every stage stays reachable after
// the analysis, so the user can go back and see what each step did.
// Below lg it collapses to a single <select> so it doesn't push the content
// a full screen down on a phone.
//
// On desktop the eight stages are a rail rather than a list of pills, and the
// connector carries the `scope` field from stages.js: solid through the stages
// both pipelines share, crimson through the ChroLumi-only ones, dashed through
// the experimental undertone step. That information used to sit unread inside
// an eyebrow string.

const RAIL = {
  "shared by both pipelines": {
    line: "bg-line",
    node: "border-ink-soft",
    dashed: false,
  },
  "ChroLumi only": {
    line: "bg-accent/35",
    node: "border-accent",
    dashed: false,
  },
  experimental: {
    line: "bg-transparent",
    node: "border-warm-ink",
    dashed: true,
  },
};

const railOf = (scope) => RAIL[scope] ?? RAIL["shared by both pipelines"];

// warm-ink at the weight a 2px dashed rule needs to read as a rule and not as
// a row of dark ticks. Shared by the rail and its key so they cannot drift.
const DASH = "#B99A3F";
const dashes = (dir) => ({
  backgroundImage: `repeating-linear-gradient(to ${dir},${DASH} 0 3px,transparent 3px 7px)`,
});

export default function ResultsSidebar({ stage, onPick, preview, onReset }) {
  const stepIdx = STEPS.findIndex((s) => s.id === stage);

  return (
    <nav
      aria-label="Result stages"
      className="w-full lg:w-[270px] lg:shrink-0 flex flex-col gap-4 lg:sticky lg:top-6"
    >
      <Card className="p-3.5 flex gap-3 items-center">
        {preview && (
          <img
            src={preview}
            alt=""
            className="w-16 h-16 rounded-readout object-cover shrink-0 border border-rule"
          />
        )}
        <button
          onClick={onReset}
          className="rounded-full bg-white border border-line text-accent text-fine px-4 min-h-[36px] hover:border-accent transition-colors duration-200"
        >
          New analysis
        </button>
      </Card>

      {/* Phone / tablet: one dropdown + progress ticks. */}
      <div className="lg:hidden flex flex-col gap-2">
        <label className="font-mono text-data text-ink-mid" htmlFor="stage-select">
          {stepIdx >= 0 ? `Stage ${stepIdx + 1} of ${STEPS.length}` : "Stage"}
        </label>
        <select
          id="stage-select"
          value={stage}
          onChange={(e) => onPick(e.target.value)}
          className="min-h-[48px] rounded-readout border border-line bg-white px-3 text-base font-semibold"
        >
          <option value="summary">Summary</option>
          <optgroup label="ChroLumi pipeline">
            {STEPS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.n} {s.label}
              </option>
            ))}
          </optgroup>
          <optgroup label="Compare">
            <option value="compare">Baseline vs ChroLumi</option>
          </optgroup>
        </select>
        <div
          className="grid gap-1"
          style={{ gridTemplateColumns: `repeat(${STEPS.length}, minmax(0,1fr))` }}
        >
          {STEPS.map((s, i) => (
            <div
              key={s.id}
              className={`h-1 rounded-sm transition-colors duration-300 ${
                i <= stepIdx ? "bg-accent" : "bg-line"
              }`}
            />
          ))}
        </div>
      </div>

      {/* Desktop: the spine. */}
      <Card className="hidden lg:block p-2.5">
        <FlatButton active={stage === "summary"} onClick={() => onPick("summary")}>
          Summary
        </FlatButton>

        <GroupLabel>ChroLumi pipeline</GroupLabel>
        <ol className="relative">
          {STEPS.map((s, i) => (
            <RailItem
              key={s.id}
              step={s}
              active={stage === s.id}
              first={i === 0}
              last={i === STEPS.length - 1}
              onClick={() => onPick(s.id)}
            />
          ))}
        </ol>

        <GroupLabel>Compare</GroupLabel>
        <FlatButton active={stage === "compare"} onClick={() => onPick("compare")}>
          Baseline vs ChroLumi
        </FlatButton>
      </Card>

      <ul className="hidden lg:flex flex-col gap-1.5 px-3 text-data text-ink-mid">
        <Key className="bg-line">Both pipelines</Key>
        <Key className="bg-accent/35">ChroLumi only</Key>
        <Key dashed>Experimental</Key>
      </ul>
    </nav>
  );
}

function Key({ className = "", dashed, children }) {
  return (
    <li className="flex items-center gap-2.5">
      <span
        aria-hidden="true"
        className={`w-5 h-[2px] shrink-0 ${className}`}
        style={dashed ? dashes("right") : undefined}
      />
      {children}
    </li>
  );
}

function GroupLabel({ children }) {
  return <div className="text-data text-ink-mid px-3 pt-4 pb-2">{children}</div>;
}

function RailItem({ step, active, first, last, onClick }) {
  const rail = railOf(step.scope);

  return (
    <li className="relative">
      {/* Connector. Stops at the node on the first and last stage so the rail
          doesn't run off the ends of the list. */}
      <span
        aria-hidden="true"
        className={`absolute left-[15px] w-[2px] ${rail.line} ${
          first ? "top-1/2 bottom-0" : last ? "top-0 bottom-1/2" : "top-0 bottom-0"
        }`}
        style={rail.dashed ? dashes("bottom") : undefined}
      />
      <button
        onClick={onClick}
        aria-current={active ? "step" : undefined}
        className={`relative flex items-center gap-3 w-full min-h-[44px] pl-9 pr-3 py-2 rounded-[10px] text-left text-fine transition-colors duration-200 ${
          active ? "bg-accent text-white" : "text-ink hover:bg-blush-soft"
        }`}
      >
        <span
          aria-hidden="true"
          className={`absolute left-[9px] w-3.5 h-3.5 rounded-full border-2 transition-colors duration-200 ${
            active ? "bg-white border-white" : `bg-white ${rail.node}`
          }`}
        />
        <span className={`font-mono text-data ${active ? "text-blush" : "text-ink-mid"}`}>
          {step.n}
        </span>
        <span className="flex-1">{step.label}</span>
      </button>
    </li>
  );
}

function FlatButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={`flex items-center w-full min-h-[44px] px-3 py-2 rounded-[10px] text-left text-fine font-medium transition-colors duration-200 ${
        active ? "bg-accent text-white" : "text-ink hover:bg-blush-soft"
      }`}
    >
      {children}
    </button>
  );
}
