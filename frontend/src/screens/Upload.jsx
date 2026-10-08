import { useRef, useState } from "react";
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
    <main className="max-w-5xl mx-auto px-8 py-12">
      <span className="font-mono text-[11px] tracking-widest text-accent">
        SKIN TONE &amp; UNDERTONE ANALYSIS
      </span>

      <h1 className="font-display text-5xl leading-tight mt-4">
        Discover your
        <br />
        <span className="italic text-accent">skin tone &amp; undertone</span>
      </h1>

      <p className="max-w-md text-[15px] leading-relaxed text-ink-soft mt-5">
        Upload a clear facial photo. Your image is analyzed twice — once by the baseline
        pipeline and once by HueView — so you can see where the two disagree.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-[1.15fr_0.85fr] gap-10 mt-10 pt-8 border-t border-line">
        <div>
          <div
            role="button"
            tabIndex={0}
            onClick={() => input.current.click()}
            onKeyDown={(e) => e.key === "Enter" && input.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files[0]); }}
            className={`h-64 rounded-2xl bg-white border-2 border-dashed flex flex-col items-center justify-center cursor-pointer focus:outline focus:outline-2 focus:outline-accent ${
              drag ? "border-accent" : "border-line"
            }`}
          >
            <div className="w-14 h-14 rounded-xl bg-blush grid place-items-center">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                   stroke="#C9145C" strokeWidth="2" strokeLinecap="round">
                <path d="M12 17V3M12 3l-5 5M12 3l5 5M4 21h16" />
              </svg>
            </div>
            <div className="font-semibold text-[17px] mt-4">
              {busy ? "Detecting face…" : "Drop photo here"}
            </div>
            <div className="text-[13px] text-ink-soft mt-1">
              or click to browse — JPG or PNG
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
            <p className="mt-4 text-[13px] text-accent bg-white border border-line rounded-xl px-4 py-3">
              {error}
            </p>
          )}
        </div>

        <div>
          <span className="font-mono text-[11px] tracking-widest text-accent">
            SCC REFERENCE SCALE
          </span>

          <p className="text-[13px] leading-relaxed text-ink-soft mt-3">
            <b className="text-ink">SCC = {SCC_MEANING.name}</b>, the six skin tone groups this
            tool classifies into. {SCC_MEANING.basis}
          </p>

          <ul className="flex flex-col gap-2 mt-4">
            {SCC.map((s) => (
              <li key={s.id} className="flex items-center gap-3">
                <span className="w-10 h-10 rounded-lg" style={{ background: s.hex }} />
                <span>
                  <span className="block font-mono text-[11px] text-accent">{s.id}</span>
                  <span className="block text-[13px]">{s.label}</span>
                </span>
                <span className="ml-auto font-mono text-[11px] text-ink-soft">{s.mst}</span>
              </li>
            ))}
          </ul>

          <p className="text-[11px] text-ink-soft mt-2">
            Swatches are indicative colours for each group, not the MST reference patches.
          </p>

          <div className="flex gap-2 mt-5">
            <span className="rounded-full bg-warm text-warm-ink font-mono text-[11px] px-3 py-1">Warm</span>
            <span className="rounded-full bg-neutraltone text-neutraltone-ink font-mono text-[11px] px-3 py-1">Neutral</span>
            <span className="rounded-full bg-cool text-cool-ink font-mono text-[11px] px-3 py-1">Cool</span>
          </div>
        </div>
      </div>
    </main>
  );
}
