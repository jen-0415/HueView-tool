import Card, { PanelHead } from "../components/Card";

export default function Confirm({ preview, detection, error, onRun, onBack }) {
  return (
    <main className="max-w-[1020px] mx-auto px-4 sm:px-8 py-12">
      <h2 className="font-display text-head font-semibold">Face detected</h2>
      <p className="text-base text-ink-mid mt-2">
        Check the crop before running both pipelines on it.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 mt-8">
        <Card>
          <PanelHead title="Original upload" />
          <div className="p-6 flex justify-center">
            <img
              src={preview}
              alt="The photo you uploaded"
              className="w-52 h-52 object-cover rounded-readout border border-rule"
            />
          </div>
        </Card>

        <Card tier="readout">
          <PanelHead title="Detected and cropped" note="This is what both models see." />
          <div className="p-6 flex justify-center">
            <img
              src={detection.crop_preview || preview}
              alt="The detected face, cropped and resized"
              className="w-52 h-52 object-cover rounded-readout outline outline-2 outline-offset-2 outline-accent"
            />
          </div>
        </Card>
      </div>

      <Card className="mt-5">
        <dl className="grid grid-cols-2 divide-x divide-line-soft">
          <Stat label="Crop resolution" value="224 × 224 px" />
          <Stat label="Key points found" value={`${detection.landmarks_found} / 5`} accent />
        </dl>
      </Card>

      {detection.upscaled && (
        <p className="mt-4 text-base text-warm-ink bg-warm rounded-readout px-5 py-3.5">
          This face was smaller than 224 px and has been upscaled. Results may be less reliable.
        </p>
      )}

      {error && (
        <p
          role="alert"
          className="mt-4 text-base text-accent bg-white border border-line rounded-readout px-5 py-3.5"
        >
          {error}
        </p>
      )}

      <div className="flex items-center gap-5 mt-8 flex-wrap">
        <button
          onClick={onRun}
          className="rounded-full px-9 py-4 text-white font-semibold bg-accent hover:bg-accent-light transition-colors duration-200"
        >
          Run both analyses
        </button>
        <button
          onClick={onBack}
          className="text-fine text-ink-mid underline decoration-line underline-offset-4 hover:text-accent hover:decoration-accent transition-colors duration-200"
        >
          Use a different photo
        </button>
      </div>
    </main>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div className="px-6 py-5 text-center">
      <dt className="text-fine text-ink-mid">{label}</dt>
      <dd className={`font-mono text-title mt-1.5 ${accent ? "text-accent" : "text-ink"}`}>
        {value}
      </dd>
    </div>
  );
}
