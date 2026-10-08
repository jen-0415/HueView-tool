const TABS = [
  { id: "analyze", label: "Analyze a photo" },
  { id: "evaluation", label: "Evaluation Results" },
];

export default function Header({ tab, onTab }) {
  return (
    <header className="bg-blush-soft border-b border-line px-4 sm:px-8 py-4 flex items-center gap-6 flex-wrap">
      <div>
        <span className="font-display italic text-2xl text-ink">Hue</span>
        <span className="font-display text-2xl text-accent">View</span>
      </div>
      {onTab && (
        <nav className="flex gap-1" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => onTab(t.id)}
              aria-current={tab === t.id ? "page" : undefined}
              className={`rounded-full px-4 min-h-[40px] text-sm ${
                tab === t.id
                  ? "bg-accent text-white"
                  : "text-ink-soft hover:text-accent hover:bg-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      )}
    </header>
  );
}
