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

const pct = (v, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
const signed = (v) => (v == null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(2)} pts`);

// Pages, in the order the study reports them.
const PAGES = [
  { id: "baseline", n: "01", group: "Lighting conditions", label: "Baseline (global RGB)" },
  { id: "hueview", n: "02", group: "Lighting conditions", label: "HueView" },
  // Not a numbered page: it summarises 01 and 02 side by side.
  { id: "compare", n: "⇄", group: "Lighting conditions", label: "Lighting Conditions Summary", extra: true },
  { id: "regions", n: "03", group: "Facial regions", label: "HueView by region" },
  { id: "tests", n: "04", group: "Comparison", label: "Significance tests" },
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
        {page === "compare" && (
          <ComparePage page={PAGES[2]} baseline={data.baseline_lighting} hueview={data.hueview_lighting} />
        )}
        {page === "regions" && <RegionPage page={PAGES[3]} block={data.regions} />}
        {page === "tests" && <SignificancePage page={PAGES[4]} block={data.significance} />}

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
          {page.extra
            ? `Summary · ${page.group}`
            : `${page.n} of ${String(PAGES.filter((q) => !q.extra).length).padStart(2, "0")} · ${page.group}`}
        </div>
        <h2 className="font-display text-[32px] font-medium mt-1.5">{title}</h2>
        <p className="text-sm leading-relaxed text-[#3A2A2E] mt-1 text-justify">{description}</p>
      </div>
      {findings && (
        <div className="bg-blush-soft border border-line rounded-[14px] px-5 py-4">
          <div className="font-mono text-[11px] tracking-widest uppercase text-accent">Key findings</div>
          <div className="mt-2 text-[15px] leading-relaxed space-y-2 text-justify">{findings}</div>
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

  const answer = (
    <>
      <p>
        The {model} is most accurate under <b>{bestAcc.Illumination}</b> illumination (
        {pct(bestAcc.Accuracy)} accuracy, {pct(bestAcc["Macro F1-Score"])} macro F1) and least accurate
        under <b>{worstAcc.Illumination}</b> illumination ({pct(worstAcc.Accuracy)} accuracy,{" "}
        {pct(worstAcc["Macro F1-Score"])} macro F1).
      </p>
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

/* ------------------------------------------------------ comparison page */

const signedPts = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)} pts`;

// Baseline and HueView next to each other in every illumination bin, from the
// same Table 19 rows the two lighting pages show. Difference = HueView − Baseline;
// it is descriptive only (significance is tested in significance.py).
function ComparePage({ page, baseline, hueview }) {
  const bins = baseline.metrics.map((r) => r.Illumination);
  const hvBy = Object.fromEntries(hueview.metrics.map((r) => [r.Illumination, r]));
  const pairs = baseline.metrics
    .filter((b) => hvBy[b.Illumination])
    .map((b) => ({ bin: b.Illumination, n: b.n, b, h: hvBy[b.Illumination] }));

  // Lead with where HueView is strongest relative to the Baseline; every number
  // is still in the table below.
  const gap = (p, key) => p.h[key] - p.b[key];
  const bestF1 = [...pairs].sort((x, z) => gap(z, "Macro F1-Score") - gap(x, "Macro F1-Score"))[0];
  const bestRecall = [...pairs].sort((x, z) => gap(z, "Macro Recall") - gap(x, "Macro Recall"))[0];
  const ahead = (p, key) => gap(p, key) > 0;

  const answer = (
    <>
      <p>
        The two models show different strengths across lighting conditions.{" "}
        {ahead(bestF1, "Macro F1-Score") ? (
          <>
            Under <b>{bestF1.bin.toLowerCase()}</b> lighting, HueView reaches a higher macro F1 (
            {pct(bestF1.h["Macro F1-Score"])} vs {pct(bestF1.b["Macro F1-Score"])})
          </>
        ) : (
          <>
            HueView&apos;s macro F1 is closest to the Baseline&apos;s under{" "}
            <b>{bestF1.bin.toLowerCase()}</b> lighting ({pct(bestF1.h["Macro F1-Score"])} vs{" "}
            {pct(bestF1.b["Macro F1-Score"])})
          </>
        )}
        {ahead(bestRecall, "Macro Recall") && (
          <>
            {" "}and {bestRecall === bestF1 ? "a higher macro recall" : (
              <>a higher macro recall under <b>{bestRecall.bin.toLowerCase()}</b> lighting</>
            )} (
            {pct(bestRecall.h["Macro Recall"])} vs {pct(bestRecall.b["Macro Recall"])})
          </>
        )}
        , while the Baseline&apos;s strengths lie in accuracy and precision.
      </p>
      <p className="text-[13px] text-ink-soft">
        The full comparison for every metric and lighting group is in the table below.
      </p>
    </>
  );

  return (
    <Page
      page={page}
      title="Lighting Conditions Summary"
      description="The Baseline (01) and HueView (02) side by side on the same test images in each lighting group. The difference column is HueView minus the Baseline, in percentage points."
      findings={answer}
    >
      <div className="flex flex-col gap-2">
        <SubHead note="Precision, recall and F1 are macro-averaged over the six SCC classes. The higher value in each row is bold.">
          Performance by illumination bin
        </SubHead>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-blush">
                <th className={th}>Illumination</th>
                <th className={th}>Metric</th>
                <th className={thR}>Baseline</th>
                <th className={thR}>HueView</th>
                <th className={thR}>Difference</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((p) =>
                METRICS.map((m, i) => {
                  const a = p.b[m.key];
                  const c = p.h[m.key];
                  const d = c - a;
                  return (
                    <tr
                      key={`${p.bin}-${m.key}`}
                      className={`border-t ${i === 0 ? "border-line" : "border-line-soft"}`}
                    >
                      {i === 0 && (
                        <td rowSpan={METRICS.length} className="px-3 py-2 align-top">
                          {p.bin}
                          <div className="font-mono text-[11px] text-ink-soft">
                            {p.n.toLocaleString()} images
                          </div>
                        </td>
                      )}
                      <td className="px-3 py-2 text-ink-soft">{m.label}</td>
                      <td className={`px-3 py-2 text-right font-mono ${a > c ? "font-semibold" : ""}`}>
                        {pct(a)}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${c > a ? "font-semibold text-accent" : ""}`}>
                        {pct(c)}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${d >= 0 ? "text-ok" : "text-ink-soft"}`}>
                        {signedPts(d)}
                      </td>
                    </tr>
                  );
                }),
              )}
            </tbody>
          </table>
        </div>
      </div>
      <div className="flex flex-col gap-2">
        <SubHead note="Rows are the true SCC, columns the predicted SCC; darker means a larger share of that true class.">
          Confusion matrices, side by side
        </SubHead>
        <div className="flex flex-col gap-5">
          {bins.map((bin) => (
            <div key={bin} className="flex flex-wrap gap-6">
              {baseline.confusion[bin] && (
                <Confusion title={`Baseline · ${bin} illumination`} matrix={baseline.confusion[bin]} />
              )}
              {hueview.confusion[bin] && (
                <Confusion title={`HueView · ${bin} illumination`} matrix={hueview.confusion[bin]} />
              )}
            </div>
          ))}
        </div>
      </div>
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

/* ------------------------------------------------------ significance page */

const METRIC_LABEL = Object.fromEntries(METRICS.map((m) => [m.short, m.label]));

function Decision({ reject }) {
  return reject ? (
    <span className="inline-block rounded-full bg-accent text-white text-[11px] px-2.5 py-0.5">Reject H₀</span>
  ) : (
    <span className="inline-block rounded-full bg-blush text-ink-soft text-[11px] px-2.5 py-0.5">Fail to reject</span>
  );
}

// Compact comparison table: scores, difference, decision.
function TestTable({ rows, aLabel, bLabel, groupLabel }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-blush">
            {groupLabel && <th className={th}>{groupLabel}</th>}
            <th className={th}>Metric</th>
            <th className={thR}>{aLabel}</th>
            <th className={thR}>{bLabel}</th>
            <th className={thR}>Difference</th>
            <th className={th}>Decision</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.stratum}-${r.a}-${r.b}-${r.metric}`} className="border-t border-line-soft">
              {groupLabel && <td className="px-3 py-2">{r.group}</td>}
              <td className="px-3 py-2">{METRIC_LABEL[r.metric] ?? r.metric}</td>
              <td className="px-3 py-2 text-right font-mono">{pct(r.score_a)}</td>
              <td className="px-3 py-2 text-right font-mono">{pct(r.score_b)}</td>
              <td className="px-3 py-2 text-right font-mono">{signed(r.score_b - r.score_a)}</td>
              <td className="px-3 py-2"><Decision reject={r.reject_h0} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HypothesisCard({ code, statement, decisions }) {
  return (
    <div className="border border-line-soft rounded-xl p-4">
      <div className="font-mono text-[12px] font-semibold text-accent">{code}</div>
      <p className="text-[13px] mt-1">{statement}</p>
      <ul className="mt-3 space-y-1.5">
        {decisions.map((d) => (
          <li key={d.hypothesis} className="flex items-center justify-between gap-2 text-[13px]">
            <span>{d.hypothesis.split("—")[1]?.trim()}</span>
            <Decision reject={d.conclusion.startsWith("Reject")} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function SignificancePage({ page, block }) {
  const dec = (h) => block.decisions.filter((d) => d.hypothesis.startsWith(h));
  // "Accuracy, Precision" for the rejected / not-rejected metrics of one hypothesis.
  const split = (h) => {
    const name = (d) => d.hypothesis.split("—")[1]?.trim().replace("Macro ", "");
    const ds = dec(h);
    return {
      yes: ds.filter((d) => d.conclusion.startsWith("Reject")).map(name),
      no: ds.filter((d) => !d.conclusion.startsWith("Reject")).map(name),
    };
  };
  const line = (h) => {
    const { yes, no } = split(h);
    if (!yes.length) return "no significant difference in any metric.";
    return `significant difference in ${yes.join(", ")}${no.length ? `; none in ${no.join(", ")}` : ""}.`;
  };

  const answer = (
    <>
      <p><b>Overall (H₀3):</b> {line("H03")}</p>
      <p><b>By lighting (H₀1):</b> {line("H01")}</p>
      <p><b>Across facial regions (H₀2):</b> {line("H02")}</p>
    </>
  );

  return (
    <Page
      page={page}
      title="Significance tests"
      description="Whether the differences between the Baseline and HueView, and among HueView's facial regions, are statistically significant."
      findings={answer}
    >
      <div className="grid md:grid-cols-3 gap-4">
        <HypothesisCard
          code="H₀1"
          statement="No difference between the Baseline and HueView within each lighting condition."
          decisions={dec("H01")}
        />
        <HypothesisCard
          code="H₀2"
          statement="No difference in HueView's performance among the six facial region configurations."
          decisions={dec("H02")}
        />
        <HypothesisCard
          code="H₀3"
          statement="No difference in overall performance between the Baseline and HueView."
          decisions={dec("H03")}
        />
      </div>

      <div className="flex flex-col gap-2">
        <SubHead note="Full test set. Difference = HueView − Baseline.">Baseline vs HueView</SubHead>
        <TestTable rows={block.h03} aLabel="Baseline" bLabel="HueView" />
      </div>

      <More title="By lighting condition">
        <TestTable
          rows={block.h01.map((r) => ({ ...r, group: r.stratum }))}
          aLabel="Baseline"
          bLabel="HueView"
          groupLabel="Lighting"
        />
      </More>
      <More title="Region pairs">
        <TestTable
          rows={block.h02_pairwise.map((r) => ({ ...r, group: `${r.a} vs ${r.b}` }))}
          aLabel="Region A"
          bLabel="Region B"
          groupLabel="Pair"
        />
      </More>

      <p className="text-[12px] text-ink-soft">
        Accuracy: McNemar&apos;s test (mid-p). Precision, recall and F1: paired bootstrap with BCa
        intervals. Bonferroni-adjusted α per hypothesis.
      </p>
    </Page>
  );
}
