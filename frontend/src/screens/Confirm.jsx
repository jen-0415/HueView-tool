import Card from "../components/Card";

export default function Confirm({ preview, detection, error, onRun, onBack }) {
  const pct = Math.round(detection.confidence * 100);

  return (
    <main className="max-w-5xl mx-auto px-8 py-12">
      <h2 className="font-display text-4xl">
        Face detected — <span className="italic text-accent">confirm &amp; proceed</span>
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-8">
        <Card>
          <div className="px-5 py-3 border-b border-line-soft text-sm">Original upload</div>
          <div className="p-6 flex justify-center">
            <img src={preview} alt="" className="w-52 h-52 object-cover rounded-lg" />
          </div>
        </Card>

        <Card className="border-line">
          <div className="px-5 py-3 border-b border-line-soft flex justify-between items-center">
            <span className="text-sm">Detected &amp; cropped</span>
            <span className="rounded-full bg-ok-bg text-ok font-mono text-[11px] px-3 py-1">
              {pct}% confidence
            </span>
          </div>
          <div className="p-6 flex justify-center">
            <div className="relative">
              <img
                src={detection.crop_preview || preview}
                alt=""
                className="w-52 h-52 object-cover rounded-lg border-2 border-accent"
              />
              <span className="absolute top-1.5 left-1.5 bg-black/60 text-white font-mono text-[10px] px-2 py-1">
                conf: {detection.confidence.toFixed(2)}
              </span>
            </div>
          </div>
        </Card>
      </div>

      <Card className="mt-6">
        <div className="grid grid-cols-3 py-5">
          <Stat label="Detection confidence" value={`${pct}%`} className="bg-ok-bg text-ok" />
          <Stat label="Crop resolution" value="224 × 224 px" className="bg-blush-soft text-cool-ink" />
          <Stat label="Landmarks found" value={`${detection.landmarks_found} / 5`} className="bg-blush text-accent" />
        </div>
      </Card>

      {detection.upscaled && (
        <p className="mt-4 text-[13px] text-warm-ink bg-warm rounded-xl px-4 py-3">
          This face was smaller than 224 px and has been upscaled. Results may be less reliable.
        </p>
      )}

      {error && (
        <p className="mt-4 text-[13px] text-accent bg-white border border-line rounded-xl px-4 py-3">
          {error}
        </p>
      )}

      <div className="flex items-center gap-4 mt-8">
        <button
          onClick={onRun}
          className="rounded-full px-10 py-4 text-white font-semibold bg-gradient-to-r from-accent to-accent-light"
        >
          Run both analyses
        </button>
        <button onClick={onBack} className="text-sm text-ink-soft">
          Use a different photo
        </button>
      </div>
    </main>
  );
}

function Stat({ label, value, className }) {
  return (
    <div className="text-center">
      <div className="text-[13px] text-ink-soft">{label}</div>
      <div className={`inline-block rounded-lg px-4 py-2 mt-2 font-mono text-[15px] ${className}`}>
        {value}
      </div>
    </div>
  );
}
