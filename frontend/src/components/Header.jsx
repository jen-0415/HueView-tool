import { TABS } from "../view";

export default function Header({ tab, onTab }) {
  return (
    <header className="bg-blush-soft border-b border-line px-4 sm:px-8 py-4 flex items-center gap-x-6 gap-y-3 flex-wrap">
      <div>
        <span className="font-display italic text-2xl text-ink">Hue</span>
        <span className="font-display text-2xl text-accent">View</span>
      </div>
      {onTab && (
        <nav className="flex flex-1 items-center gap-1" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => onTab(t.id)}
              aria-current={tab === t.id ? "page" : undefined}
              className={`rounded-full px-4 min-h-[40px] text-sm ${t.right ? "ml-auto" : ""} ${
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
