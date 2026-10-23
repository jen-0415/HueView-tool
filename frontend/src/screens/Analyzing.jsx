// Loading screen while the backend analyzes the photo. The SSR intermediate
// images are not shown here; they appear in the Results screen's
// lighting-correction stage.
export default function Analyzing() {
  return (
    <main className="min-h-[60vh] grid place-items-center px-8">
      <div className="flex flex-col items-center gap-5">
        <div className="w-14 h-14 rounded-full border-[5px] border-line-soft border-t-accent animate-spin" />
        <h2 className="font-display italic text-head font-normal">Analyzing photo</h2>
        <p className="text-fine text-ink-mid" role="status">
          This may take a few seconds…
        </p>
      </div>
    </main>
  );
}
