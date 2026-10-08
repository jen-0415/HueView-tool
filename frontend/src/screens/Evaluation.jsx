import { useEffect, useState } from "react";
import Card from "../components/Card";
import { getEvaluation } from "../api";

// Evaluation results from the saved result files (GET /api/evaluation): the
// Baseline and HueView under each lighting level, HueView per facial region,
// and the significance tests. Numbers are never computed here -- this screen
// only lays out what evaluate.py, appendix_tables.py and significance.py wrote,
// so it always matches the tables in Chapter 4 and the appendices.
// One page at a time, picked from the sidebar, like the analysis Results.

const METRICS = [
  { key: "Accuracy", short: "accuracy", label: "Accuracy" },
  { key: "Macro Precision", short: "precision_macro", label: "Precision" },
  { key: "Macro Recall", short: "recall_macro", label: "Recall" },
  { key: "Macro F1-Score", short: "f1_macro", label: "F1-Score" },
];
const SMALL_BIN = 200;

// Pages, in the order the study reports them.
const PAGES = [
  { id: "baseline", n: "01", group: "Lighting conditions", label: "Baseline (global RGB)" },
  { id: "hueview", n: "02", group: "Lighting conditions", label: "HueView" },
  { id: "regions", n: "03", group: "Facial regions", label: "HueView by region" },
  { id: "tests", n: "04", group: "Comparison", label: "Significance tests" },
];

const pct = (v, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
const signed = (v) => (v == null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(2)} pts`);
const pval = (p) => (p == null ? "—" : p < 0.0001 ? p.toExponential(2) : p.toFixed(4));
const stamp = (t) => t.replace("T", " ");

export default function Evaluation() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(PAGES[0].id);

  useEffect(() => {
    getEvaluation().then(setData).catch((e) => setError(e.message));
  }, []);

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
        {data.warnings.length > 0 && (
          <div className="border-2 border-accent bg-white rounded-2xl px-5 py-4" role="alert">
            <div className="font-mono text-[11px] tracking-widest uppercase text-accent">Results not final</div>
            <ul className="mt-2 text-sm list-disc pl-5 space-y-1">
              {data.warnings.map((w) => <li key={w}>{w}</li>)}
            </ul>
          </div>
        )}

        {page === "baseline" && (
          <LightingPage
            page={PAGES[0]}
            title="Baseline under different lighting"
            description="Skin tone classification by the Baseline (EfficientNetB0 with global RGB averaging) on test images grouped into low, medium and high lighting."
            model="Baseline"
            block={data.baseline_lighting}
            tables="12–14"
          />
        )}
        {page === "hueview" && (
          <LightingPage
            page={PAGES[1]}
            title="HueView under different lighting"
            description="Skin tone classification by HueView (EfficientNetB0 with Single-Scale Retinex, regional facial analysis and CIELAB; Full Face result) on the same test images and lighting groups."
            model="HueView"
            block={data.hueview_lighting}
            tables="15–17"
          />
        )}
        {page === "regions" && <RegionPage page={PAGES[2]} block={data.regions} />}
        {page === "tests" && <SignificancePage page={PAGES[3]} block={data.significance} />}

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
        <div className="text-[11px] text-ink-soft mt-1 leading-relaxed">
          Predictions saved {stamp(data.generated.record)}
          <br />
          Significance tests run {stamp(data.generated.significance)}
        </div>
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
          {["TP", "TN", "FP", "FN", "Support"].map((k) => <th key={k} className={thR}>{k}</th>)}
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
            {["TP", "TN", "FP", "FN", "Support"].map((k) => (
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

function LightingPage({ page, title, description, model, block, tables }) {
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
          Performance by illumination bin (Table 19)
        </SubHead>
        <MetricTable rows={rows} groupKey="Illumination" groupLabel="Illumination" />
      </div>
      <div className="flex flex-col gap-2">
        <SubHead note="Rows are the true SCC, columns the predicted SCC; darker means a larger share of that true class.">
          Confusion matrices (Tables {tables})
        </SubHead>
        <div className="flex flex-wrap gap-6">
          {Object.entries(block.confusion).map(([bin, m]) => (
            <Confusion key={bin} title={`${bin} illumination`} matrix={m} />
          ))}
        </div>
      </div>
      <More title="Per-class results (Table 18)">
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
          The <b>Full Face</b> configuration, the majority vote of the five regions, reaches{" "}
          {pct(full.Accuracy)} accuracy and {pct(full["Macro F1-Score"])} macro F1.
        </p>
      )}
    </>
  );

  return (
    <Page
      page={page}
      title="HueView by facial region"
      description="HueView classifying from each facial region on its own (forehead, left cheek, right cheek, jawline, nose bridge) and from the Full Face, the majority vote of the five."
      findings={answer}
    >
      <div className="flex flex-col gap-2">
        <SubHead note="Each region was evaluated on the same test images.">
          Performance by facial region (Table 28)
        </SubHead>
        <MetricTable rows={rows} groupKey="Region" groupLabel="Region" />
      </div>
      <div className="flex flex-col gap-2">
        <SubHead>Confusion matrices (Tables 21–26)</SubHead>
        <div className="flex flex-wrap gap-6">
          {Object.entries(block.confusion).map(([region, m]) => (
            <Confusion key={region} title={region} matrix={m} />
          ))}
        </div>
      </div>
      <More title="Per-class results (Table 27)">
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

// Statistic column: McNemar's b/c + mid-p for accuracy, BCa CI for macro metrics.
function evidence(r) {
  if (r.metric === "accuracy") return `${r.statistic}; mid-p = ${pval(r.p_value)}`;
  const lvl = r.ci_level ? `${(r.ci_level * 100).toFixed(2)}%` : "";
  return `${lvl} BCa CI [${signed(r.ci_low)}, ${signed(r.ci_high)}]`;
}

function TestTable({ rows, aLabel, bLabel, showStratum = true }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-blush">
            {showStratum && <th className={th}>Condition</th>}
            <th className={th}>Metric</th>
            <th className={thR}>{aLabel}</th>
            <th className={thR}>{bLabel}</th>
            <th className={thR}>Difference</th>
            <th className={th}>Test result</th>
            <th className={th}>Alpha</th>
            <th className={th}>Decision</th>
            <th className={th}>Favours</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.stratum}-${r.a}-${r.b}-${r.metric}`} className="border-t border-line-soft align-top">
              {showStratum && <td className="px-3 py-2">{r.stratum}</td>}
              <td className="px-3 py-2">{METRIC_LABEL[r.metric] ?? r.metric}</td>
              <td className="px-3 py-2 text-right font-mono">{pct(r.score_a)}</td>
              <td className="px-3 py-2 text-right font-mono">{pct(r.score_b)}</td>
              <td className="px-3 py-2 text-right font-mono">{signed(r.score_b - r.score_a)}</td>
              <td className="px-3 py-2 font-mono text-[11px] text-ink-soft">{evidence(r)}</td>
              <td className="px-3 py-2 font-mono text-[11px]">{r.alpha?.toFixed(4)}</td>
              <td className="px-3 py-2"><Decision reject={r.reject_h0} /></td>
              <td className="px-3 py-2">{r.reject_h0 ? r.favours : "—"}</td>
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
            <span className="flex items-center gap-2">
              {d.bins_rejected && d.bins_rejected !== "none" && d.bins_rejected !== "full test set" && (
                <span className="text-[11px] text-ink-soft">{d.bins_rejected}</span>
              )}
              <Decision reject={d.conclusion.startsWith("Reject")} />
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function SignificancePage({ page, block }) {
  const dec = (h) => block.decisions.filter((d) => d.hypothesis.startsWith(h));
  const h03 = block.h03;
  const sigH03 = h03.filter((r) => r.reject_h0);
  const omni = block.h02_omnibus.find((r) => r.metric.startsWith("Accuracy"));

  const answer = (
    <>
      <p>
        <b>Overall (H₀3):</b>{" "}
        {sigH03.length === 0
          ? "no metric differs significantly between the Baseline and HueView."
          : `${sigH03.map((r) => `${METRIC_LABEL[r.metric]} (favours ${r.favours})`).join(", ")} ${sigH03.length === 1 ? "differs" : "differ"} significantly; ${h03.filter((r) => !r.reject_h0).map((r) => METRIC_LABEL[r.metric]).join(", ") || "no other metric"} ${h03.length - sigH03.length === 1 ? "does" : "do"} not.`}
      </p>
      <p>
        <b>By illumination (H₀1):</b>{" "}
        {block.h01.filter((r) => r.reject_h0).length === 0
          ? "no significant difference in any bin."
          : block.h01
              .filter((r) => r.reject_h0)
              .map((r) => `${METRIC_LABEL[r.metric]} under ${r.stratum} (favours ${r.favours})`)
              .join("; ") + "."}
      </p>
      {omni && (
        <p>
          <b>Across regions (H₀2):</b> Cochran&apos;s {omni.statistic}, p = {pval(omni.p_value)} —{" "}
          {omni.reject_h0 ? "accuracy differs among the six region configurations." : "no significant difference in accuracy among regions."}{" "}
          Pairwise results are below.
        </p>
      )}
    </>
  );

  return (
    <Page
      page={page}
      title="Significance tests"
      description="Whether the differences between the Baseline and HueView (overall and per lighting level), and among HueView's facial regions, are statistically significant."
      findings={answer}
    >
      <div className="grid md:grid-cols-3 gap-4">
        <HypothesisCard
          code="H₀1"
          statement="No significant difference between the Baseline and HueView within each illumination condition."
          decisions={dec("H01")}
        />
        <HypothesisCard
          code="H₀2"
          statement="No significant difference in HueView's performance among the six facial region configurations."
          decisions={dec("H02")}
        />
        <HypothesisCard
          code="H₀3"
          statement="No significant difference in overall performance between the Baseline and HueView."
          decisions={dec("H03")}
        />
      </div>
      <p className="text-[12px] text-ink-soft -mt-2">
        Accuracy: McNemar&apos;s test with mid-p correction. Precision, recall, F1: paired bootstrap (B = 10,000,
        stratified by SCC) with BCa intervals; significant when the interval excludes zero. Bonferroni α: H₀1
        0.05/12, H₀2 0.05/15, H₀3 0.05/4. For H₀1 and H₀2 a metric is rejected when any bin or pair is
        significant.
      </p>

      <div className="flex flex-col gap-2">
        <SubHead note="Difference = HueView − Baseline, in percentage points.">H₀3 — full test set (Table 31.D)</SubHead>
        <TestTable rows={h03} aLabel="Baseline" bLabel="HueView" showStratum={false} />
      </div>
      <div className="flex flex-col gap-2">
        <SubHead note="Difference = HueView − Baseline, in percentage points.">H₀1 — by illumination (Table 31.A)</SubHead>
        <TestTable rows={block.h01} aLabel="Baseline" bLabel="HueView" />
      </div>
      <div className="flex flex-col gap-2">
        <SubHead>H₀2 — omnibus (Table 31.B)</SubHead>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-blush">
                <th className={th}>Metric</th>
                <th className={th}>Test</th>
                <th className={th}>Result</th>
                <th className={th}>Decision</th>
              </tr>
            </thead>
            <tbody>
              {block.h02_omnibus.map((r) => (
                <tr key={r.metric} className="border-t border-line-soft">
                  <td className="px-3 py-2">{r.metric}</td>
                  <td className="px-3 py-2">{r.test}</td>
                  <td className="px-3 py-2 font-mono text-[12px]">
                    {r.statistic}
                    {r.p_value != null && `; p = ${pval(r.p_value)}`}
                  </td>
                  <td className="px-3 py-2"><Decision reject={r.reject_h0} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <More title="H₀2 — all 15 region pairs × 4 metrics (Table 31.C; difference = Region B − Region A)">
        <TestTable
          rows={block.h02_pairwise.map((r) => ({ ...r, stratum: `${r.a} vs. ${r.b}` }))}
          aLabel="Region A"
          bLabel="Region B"
        />
      </More>
    </Page>
  );
}
