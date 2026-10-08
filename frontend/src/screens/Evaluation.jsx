import { useState } from "react";
import Card from "../components/Card";
import { useEvaluation } from "../useEvaluation";

// Evaluation results from the saved result files (GET /api/evaluation): the
// Baseline and HueView under each lighting level, and HueView per facial
// region. Numbers are never computed here -- this screen only lays out what
// evaluate.py and appendix_tables.py wrote, so it always matches the tables in
// Chapter 4 and the appendices.
// One page at a time, picked from the sidebar, like the analysis Results.

const METRICS = [
  { key: "Accuracy", short: "accuracy", label: "Accuracy" },
  { key: "Macro Precision", short: "precision_macro", label: "Precision" },
  { key: "Macro Recall", short: "recall_macro", label: "Recall" },
  { key: "Macro F1-Score", short: "f1_macro", label: "F1-Score" },
];
const SMALL_BIN = 200;

const pct = (v, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);

// Pages, in the order the study reports them.
const PAGES = [
  { id: "baseline", n: "01", group: "Lighting conditions", label: "Baseline (global RGB)" },
  { id: "hueview", n: "02", group: "Lighting conditions", label: "HueView" },
  { id: "regions", n: "03", group: "Facial regions", label: "HueView by region" },
];

export default function Evaluation() {
  const { data, error } = useEvaluation();
  const [page, setPage] = useState(PAGES[0].id);

  if (error) {
    return (
      <main className="max-w-[1100px] mx-auto px-4 sm:px-8 py-7">
        <Card className="px-6 py-5 text-sm">
          <div className="font-semibold text-accent">The results could not be loaded.</div>
          <p className="text-ink-soft mt-1">{error}</p>
          <p className="text-ink-soft mt-1">
            This tab reads the backend&apos;s saved evaluation files, so the API server has to be running.
          </p>
        </Card>
      </main>
    );
  }
  if (!data) {
    return <main className="max-w-[1100px] mx-auto px-4 sm:px-8 py-7 text-ink-soft">Loading results…</main>;
  }

  const idx = PAGES.findIndex((q) => q.id === page);
  const pick = (id) => {
    setPage(id);
    window.scrollTo({ top: 0 });
  };

  return (
    <main className="max-w-[1360px] mx-auto px-4 sm:px-8 py-7 flex flex-col lg:flex-row gap-7 items-start">
      <EvaluationSidebar page={page} onPick={pick} data={data} />

      <section className="flex-1 min-w-0 w-full flex flex-col gap-5">
        {page === "baseline" && (
          <LightingPage
            page={PAGES[0]}
            title="Baseline under different lighting"
            description="Skin tone classification by the Baseline (EfficientNetB0 with global RGB averaging) on test images grouped into low, medium and high lighting."
            model="Baseline"
            block={data.baseline_lighting}
          />
        )}
        {page === "hueview" && (
          <LightingPage
            page={PAGES[1]}
            title="HueView under different lighting"
            description="Skin tone classification by HueView (EfficientNetB0 with Single-Scale Retinex, regional facial analysis and CIELAB; final result, the majority vote of its six configurations) on the same test images and lighting groups."
            model="HueView"
            block={data.hueview_lighting}
          />
        )}
        {page === "regions" && <RegionPage page={PAGES[2]} block={data.regions} />}

        <div className="flex justify-between gap-3 flex-wrap border-t border-line pt-4">
          <div>
            {idx > 0 && (
              <button
                onClick={() => pick(PAGES[idx - 1].id)}
                className="rounded-full bg-white border border-line text-accent text-sm px-4 min-h-[44px]"
              >
                ← {PAGES[idx - 1].label}
              </button>
            )}
          </div>
          <div>
            {idx < PAGES.length - 1 && (
              <button
                onClick={() => pick(PAGES[idx + 1].id)}
                className="rounded-full bg-accent text-white text-sm font-semibold px-5 min-h-[44px]"
              >
                {PAGES[idx + 1].label} →
              </button>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

/* ----------------------------------------------------------------- sidebar */

// Same look as the analysis ResultsSidebar: grouped list on desktop, a single
// <select> below lg.
function EvaluationSidebar({ page, onPick, data }) {
  const groups = [...new Set(PAGES.map((q) => q.group))];
  return (
    <nav
      aria-label="Evaluation results"
      className="w-full lg:w-[270px] lg:shrink-0 flex flex-col gap-4 lg:sticky lg:top-6"
    >
      <Card className="p-4">
        <div className="font-mono text-[11px] tracking-widest uppercase text-ink-soft">Test set</div>
        <div className="font-display text-2xl mt-0.5">{data.n_test.toLocaleString()} images</div>
      </Card>

      <div className="lg:hidden">
        <select
          aria-label="Result page"
          value={page}
          onChange={(e) => onPick(e.target.value)}
          className="w-full min-h-[48px] rounded-xl border border-line bg-white px-3 text-[15px] font-semibold"
        >
          {groups.map((g) => (
            <optgroup key={g} label={g}>
              {PAGES.filter((q) => q.group === g).map((q) => (
                <option key={q.id} value={q.id}>
                  {q.n} {q.label}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>

      <Card className="hidden lg:flex p-2.5 flex-col gap-0.5">
        {groups.map((g) => (
          <div key={g} className="flex flex-col gap-0.5">
            <div className="font-mono text-[11px] tracking-widest uppercase text-ink-soft px-3 pt-3.5 pb-1.5">
              {g}
            </div>
            {PAGES.filter((q) => q.group === g).map((q) => {
              const active = page === q.id;
              return (
                <button
                  key={q.id}
                  onClick={() => onPick(q.id)}
                  aria-current={active ? "page" : undefined}
                  className={`flex items-center gap-2.5 w-full min-h-[44px] px-3 py-2 rounded-[10px] text-left text-sm ${
                    active ? "bg-accent text-white" : "text-ink hover:bg-blush-soft"
                  }`}
                >
                  <span className={`font-mono text-[11px] w-5 ${active ? "text-[#FFE3EE]" : "text-ink-soft"}`}>
                    {q.n}
                  </span>
                  <span className="flex-1">{q.label}</span>
                </button>
              );
            })}
          </div>
        ))}
      </Card>
    </nav>
  );
}

/* ------------------------------------------------------------------ layout */

function Page({ page, title, description, findings, children }) {
  return (
    <>
      <div>
        <div className="font-mono text-[11px] tracking-widest uppercase text-ink-soft">
          {page.n} of {String(PAGES.length).padStart(2, "0")} · {page.group}
        </div>
        <h2 className="font-display text-[32px] font-medium mt-1.5">{title}</h2>
        <p className="text-sm leading-relaxed text-[#3A2A2E] mt-1 max-w-3xl">{description}</p>
      </div>
      {findings && (
        <div className="bg-blush-soft border border-line rounded-[14px] px-5 py-4">
          <div className="font-mono text-[11px] tracking-widest uppercase text-accent">Key findings</div>
          <div className="mt-2 text-[15px] leading-relaxed space-y-2">{findings}</div>
        </div>
      )}
      <Card className="px-4 sm:px-6 py-5 flex flex-col gap-6 min-w-0">{children}</Card>
    </>
  );
}

function SubHead({ children, note }) {
  return (
    <div>
      <div className="font-mono text-[11px] font-semibold">{children}</div>
      {note && <p className="text-[12px] text-ink-soft mt-0.5">{note}</p>}
    </div>
  );
}

function More({ title, children }) {
  return (
    <details className="border border-line-soft rounded-xl group">
      <summary className="cursor-pointer px-4 py-3 text-[13px] text-accent select-none">{title}</summary>
      <div className="px-4 pb-4 overflow-x-auto">{children}</div>
    </details>
  );
}

const th = "px-3 py-2 text-left font-normal text-[10px] tracking-widest uppercase text-accent";
const thR = "px-3 py-2 text-right font-normal text-[10px] tracking-widest uppercase text-accent";

/* ------------------------------------------------------- metric tables */

// One row per group (illumination bin or region), four metrics with a bar each.
// The best value per metric is bold.
function MetricTable({ rows, groupKey, groupLabel }) {
  const best = Object.fromEntries(METRICS.map((m) => [m.key, Math.max(...rows.map((r) => r[m.key]))]));
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-blush">
            <th className={th}>{groupLabel}</th>
            <th className={thR}>Images</th>
            {METRICS.map((m) => <th key={m.key} className={thR}>{m.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[groupKey]} className="border-t border-line-soft">
              <td className="px-3 py-2">
                {r[groupKey]}
                {r.n < SMALL_BIN && (
                  <span className="ml-2 text-[11px] text-accent">small sample</span>
                )}
              </td>
              <td className="px-3 py-2 text-right font-mono text-ink-soft">{r.n.toLocaleString()}</td>
              {METRICS.map((m) => (
                <td key={m.key} className="px-3 py-2 text-right">
                  <div className={`font-mono ${r[m.key] === best[m.key] ? "font-semibold" : ""}`}>
                    {pct(r[m.key])}
                  </div>
                  <div className="h-1.5 bg-blush rounded-full mt-1 ml-auto w-24" aria-hidden="true">
                    <div className="h-full bg-accent rounded-full" style={{ width: `${r[m.key] * 100}%` }} />
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PerClassTable({ rows, groupKey }) {
  return (
    <table className="w-full text-xs font-mono">
      <thead>
        <tr className="bg-blush">
          <th className={th}>{groupKey}</th>
          <th className={th}>Class</th>
          {["TP", "TN", "FP", "FN"].map((k) => <th key={k} className={thR}>{k}</th>)}
          <th className={thR}>Precision</th>
          <th className={thR}>Recall</th>
          <th className={thR}>F1</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={`${r[groupKey]}-${r.Class}`} className="border-t border-line-soft">
            <td className="px-3 py-1.5">{r[groupKey]}</td>
            <td className="px-3 py-1.5">{r.Class}</td>
            {["TP", "TN", "FP", "FN"].map((k) => (
              <td key={k} className="px-3 py-1.5 text-right">{r[k]}</td>
            ))}
            <td className="px-3 py-1.5 text-right">{pct(r.Precision)}</td>
            <td className="px-3 py-1.5 text-right">{pct(r.Recall)}</td>
            <td className="px-3 py-1.5 text-right">{pct(r["F1-Score"])}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// Rows = actual SCC, columns = predicted SCC. Shade = share of the actual row.
function Confusion({ title, matrix }) {
  const total = matrix.rows.flat().reduce((a, b) => a + b, 0);
  return (
    <figure className="min-w-0 max-w-full">
      <figcaption className="text-[12px] font-semibold mb-1">
        {title} <span className="font-normal text-ink-soft">({total.toLocaleString()} images)</span>
      </figcaption>
      <div className="overflow-x-auto">
      <table className="text-[11px] font-mono border-collapse">
        <thead>
          <tr>
            <th className="px-1 py-1 text-[9px] text-ink-soft font-normal text-left">actual ↓ pred →</th>
            {matrix.labels.map((l) => (
              <th key={l} className="px-1 py-1 font-normal text-ink-soft">{l.replace("SCC-", "")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row, i) => {
            const rowSum = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={i}>
                <th className="px-1 py-1 font-normal text-ink-soft">{matrix.labels[i]}</th>
                {row.map((v, j) => {
                  const share = v / rowSum;
                  return (
                    <td
                      key={j}
                      title={`actual ${matrix.labels[i]}, predicted ${matrix.labels[j]}: ${v} (${pct(share)} of row)`}
                      className={`w-10 h-8 text-center border border-white ${i === j ? "outline outline-1 outline-accent -outline-offset-2" : ""}`}
                      style={{
                        background: `rgba(201, 20, 92, ${0.06 + share * 0.84})`,
                        color: share > 0.45 ? "#fff" : "#1B1416",
                      }}
                    >
                      {v}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
    </figure>
  );
}

/* -------------------------------------------------------- lighting pages */

function LightingPage({ page, title, description, model, block }) {
  const rows = block.metrics;
  const by = (k) => [...rows].sort((a, b) => b[k] - a[k]);
  const bestAcc = by("Accuracy")[0];
  const worstAcc = by("Accuracy").at(-1);
  const small = rows.filter((r) => r.n < SMALL_BIN);

  const answer = (
    <>
      <p>
        The {model} is most accurate under <b>{bestAcc.Illumination}</b> illumination (
        {pct(bestAcc.Accuracy)} accuracy, {pct(bestAcc["Macro F1-Score"])} macro F1) and least accurate
        under <b>{worstAcc.Illumination}</b> illumination ({pct(worstAcc.Accuracy)} accuracy,{" "}
        {pct(worstAcc["Macro F1-Score"])} macro F1).
      </p>
      {small.map((r) => (
        <p key={r.Illumination} className="text-[13px] text-ink-soft">
          The {r.Illumination} bin has only {r.n} test images, so its numbers move a lot with a few
          images; some SCC classes there have almost no examples.
        </p>
      ))}
    </>
  );

  return (
    <Page page={page} title={title} description={description} findings={answer}>
      <div className="flex flex-col gap-2">
        <SubHead note="Precision, recall and F1 are macro-averaged over the six SCC classes.">
          Performance by illumination bin
        </SubHead>
        <MetricTable rows={rows} groupKey="Illumination" groupLabel="Illumination" />
      </div>
      <div className="flex flex-col gap-2">
        <SubHead note="Rows are the true SCC, columns the predicted SCC; darker means a larger share of that true class.">
          Confusion matrices
        </SubHead>
        <div className="flex flex-wrap gap-6">
          {Object.entries(block.confusion).map(([bin, m]) => (
            <Confusion key={bin} title={`${bin} illumination`} matrix={m} />
          ))}
        </div>
      </div>
      <More title="Per-class results">
        <PerClassTable rows={block.perclass} groupKey="Illumination" />
      </More>
    </Page>
  );
}

/* ------------------------------------------------------------ region page */

function RegionPage({ page, block }) {
  const rows = block.metrics.map((r) => ({ ...r }));
  const single = rows.filter((r) => r.Region !== "Full Face");
  const bestSingle = [...single].sort((a, b) => b.Accuracy - a.Accuracy)[0];
  const worstSingle = [...single].sort((a, b) => a.Accuracy - b.Accuracy)[0];
  const full = rows.find((r) => r.Region === "Full Face");

  const answer = (
    <>
      <p>
        Among the five single regions, <b>{bestSingle.Region}</b> classifies best (
        {pct(bestSingle.Accuracy)} accuracy, {pct(bestSingle["Macro F1-Score"])} macro F1) and{" "}
        <b>{worstSingle.Region}</b> worst ({pct(worstSingle.Accuracy)} accuracy).
      </p>
      {full && (
        <p>
          The <b>Full Face</b> configuration, a separately trained classifier that sees the whole
          normalized face, reaches {pct(full.Accuracy)} accuracy and{" "}
          {pct(full["Macro F1-Score"])} macro F1.
        </p>
      )}
    </>
  );

  return (
    <Page
      page={page}
      title="HueView by facial region"
      description="HueView classifying from each facial region on its own (forehead, left cheek, right cheek, jawline, nose bridge) and from the Full Face, a separately trained classifier that sees the whole normalized face. All six vote on the final result."
      findings={answer}
    >
      <div className="flex flex-col gap-2">
        <SubHead note="Each region was evaluated on the same test images.">
          Performance by facial region
        </SubHead>
        <MetricTable rows={rows} groupKey="Region" groupLabel="Region" />
      </div>
      <div className="flex flex-col gap-2">
        <SubHead>Confusion matrices</SubHead>
        <div className="flex flex-wrap gap-6">
          {Object.entries(block.confusion).map(([region, m]) => (
            <Confusion key={region} title={region} matrix={m} />
          ))}
        </div>
      </div>
      <More title="Per-class results">
        <PerClassTable rows={block.perclass} groupKey="Region" />
      </More>
    </Page>
  );
}
