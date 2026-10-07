import Card from "../components/Card";

// Loading screen while the backend analyzes the photo. Once the SSR preview
// arrives it shows the face before and after illumination normalization, and
// the results wait for the user to click Continue.
export default function Analyzing({ ssrPreview, resultReady, onContinue }) {
  if (!ssrPreview) return <Spinner title="Analyzing photo" />;

  // The backend omits `steps` if its step preview ever stops matching the
  // real SSR output; fall back to just before/after.
  const steps = ssrPreview.steps ?? [
    { label: "Cropped face (original)", image: ssrPreview.original },
    { label: "After SSR", image: ssrPreview.ssr },
  ];

  return (
    <main className="max-w-5xl mx-auto px-8 py-12">
      <h2 className="font-display text-4xl">
        Illumination <span className="italic text-accent">normalized</span>
      </h2>
      <p className="mt-2 text-[13px] text-ink-soft">
        Single-Scale Retinex estimates the lighting (a blurred brightness map), divides it
        out, and applies the resulting gain to every colour channel. HueView&apos;s regional
        analysis runs on the corrected image; the Baseline uses the original.
      </p>

      <Card className="mt-8">
        <div className="px-5 py-3 border-b border-line-soft text-sm">Every intermediate image</div>
        <div className="p-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {steps.map((s, i) => (
            <figure key={s.label}>
              <img
                src={s.image}
                alt={s.label}
                className={`w-full aspect-square object-cover rounded-lg ${
                  i === steps.length - 1 ? "border-2 border-accent" : ""
                }`}
              />
              <figcaption className="mt-2">
                <div className="font-mono text-[11px] font-semibold">{String(i + 1).padStart(2, "0")}</div>
                <div className="text-[12px] text-ink-soft leading-snug">{s.label}</div>
              </figcaption>
            </figure>
          ))}
        </div>
      </Card>

      <div className="flex items-center gap-4 mt-8">
        <button
          onClick={onContinue}
          disabled={!resultReady}
          className="rounded-full px-10 py-4 text-white font-semibold bg-gradient-to-r from-accent to-accent-light disabled:opacity-50 disabled:cursor-wait"
        >
          {resultReady ? "Continue to results" : "Finishing analysis…"}
        </button>
        {!resultReady && (
          <div className="w-6 h-6 rounded-full border-[3px] border-[#F8DDE4] border-t-[#C9145C] animate-spin" />
        )}
      </div>
    </main>
  );
}

function Spinner({ title }) {
  return (
    <main className="min-h-[60vh] grid place-items-center px-8">
      <div className="flex flex-col items-center gap-5">
        <div className="w-14 h-14 rounded-full border-[5px] border-[#F8DDE4] border-t-[#C9145C] animate-spin" />
        <h2 className="font-display italic text-3xl">{title}</h2>
        <p className="text-[13px] text-ink-soft">This may take a few seconds…</p>
      </div>
    </main>
  );
}
