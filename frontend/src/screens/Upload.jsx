import { useRef, useState } from "react";
import Card, { PanelHead } from "../components/Card";
import { SCC, SCC_MEANING } from "../constants";

export default function Upload({ onFile, busy, error }) {
  const input = useRef(null);
  const [drag, setDrag] = useState(false);

  const take = (f) => {
    if (!f) return;
    if (!["image/jpeg", "image/png"].includes(f.type)) return;
    onFile(f);
  };

  return (
    <main className="max-w-[1120px] mx-auto px-4 sm:px-8 py-12 sm:py-16">
      <h1 className="font-display text-hero font-semibold max-w-[20ch]">
        Skin tone and undertone, measured twice
      </h1>

      <p className="text-base leading-relaxed text-ink-mid mt-5">
        Upload a clear facial photo. It is analysed once by the baseline pipeline and once by
        ChroLumi, so you can see where the two disagree and why.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-[1.15fr_0.85fr] gap-8 lg:gap-12 mt-12">
        <div>
          <div
            role="button"
            tabIndex={0}
            aria-label="Choose a photo to analyse"
            onClick={() => input.current.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                input.current.click();
              }
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              take(e.dataTransfer.files[0]);
            }}
            className={`group relative overflow-hidden rounded-panel bg-white cursor-pointer transition-colors duration-200 ${
              drag ? "border-2 border-accent" : "border-2 border-dashed border-line hover:border-accent/60"
            }`}
          >
            {/* The six clusters as a calibration strip along the top edge: the
                thing being measured, on the panel that takes the measurement. */}
            <div className="flex h-2" aria-hidden="true">
              {SCC.map((s) => (
                <span key={s.id} className="flex-1" style={{ background: s.hex }} />
              ))}
            </div>

            <div className="h-60 flex flex-col items-center justify-center text-center px-6">
              <div className="w-12 h-12 rounded-readout bg-blush grid place-items-center transition-transform duration-300 ease-settle group-hover:-translate-y-0.5">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#C9145C"
                  strokeWidth="2"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <path d="M12 17V3M12 3l-5 5M12 3l5 5M4 21h16" />
                </svg>
              </div>
              <div className="font-semibold text-lead mt-4">
                {busy ? "Detecting face…" : drag ? "Drop to analyse" : "Drop a photo here"}
              </div>
              <p className="text-fine text-ink-mid mt-1">or click to browse. JPG or PNG.</p>
            </div>
          </div>

          <input
            ref={input}
            type="file"
            accept="image/jpeg,image/png"
            className="hidden"
            onChange={(e) => take(e.target.files[0])}
          />

          {error && (
            <p
              role="alert"
              className="mt-4 text-base text-accent bg-white border border-line rounded-readout px-5 py-3.5"
            >
              {error}
            </p>
          )}

          <p className="text-fine text-ink-mid mt-5">
            The photo is sent to the analysis API, classified, and returned with every
            intermediate step so the result can be checked.
          </p>
        </div>

        <Card>
          <PanelHead title="The six SCC classes" note={SCC_MEANING.short} />

          <div className="px-5 py-5">
            <p className="text-fine leading-relaxed text-ink-mid">{SCC_MEANING.basis}</p>

            <ul className="flex flex-col mt-4 divide-y divide-line-soft">
              {SCC.map((s) => (
                <li key={s.id} className="flex items-center gap-3 py-2.5">
                  <span
                    className="w-9 h-9 rounded-lg shrink-0 border border-rule/60"
                    style={{ background: s.hex }}
                  />
                  <span className="min-w-0">
                    <span className="block font-mono text-data text-ink-mid">{s.id}</span>
                    <span className="block text-fine">{s.label}</span>
                  </span>
                  <span className="ml-auto font-mono text-data text-ink-mid shrink-0">
                    {s.mst}
                  </span>
                </li>
              ))}
            </ul>

            <p className="text-fine text-ink-mid mt-4">
              Swatches are indicative colours for each group, not the MST reference patches.
            </p>

            <div className="flex flex-wrap gap-2 mt-5 pt-5 border-t border-line-soft">
              <span className="text-fine text-ink-mid w-full mb-1">
                Undertone is reported alongside the class:
              </span>
              <span className="rounded-full bg-warm text-warm-ink text-data px-3 py-1">Warm</span>
              <span className="rounded-full bg-neutraltone text-neutraltone-ink text-data px-3 py-1">
                Neutral
              </span>
              <span className="rounded-full bg-cool text-cool-ink text-data px-3 py-1">Cool</span>
            </div>
          </div>
        </Card>
      </div>
    </main>
  );
}
