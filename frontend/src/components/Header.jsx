import { TABS } from "../view";

export default function Header({ tab, onTab }) {
  return (
    <header className="sticky top-0 z-20 bg-blush-soft/92 backdrop-blur-sm border-b border-line px-4 sm:px-8 py-3.5 flex items-center gap-x-6 gap-y-3 flex-wrap">
      {/* Mark and wordmark together. The mark is decorative here -- the name
          is right beside it as real text, so alt="" keeps a screen reader
          from announcing it twice. */}
      <div className="flex items-center gap-2.5 shrink-0">
        <img src="/chrolumi.png" alt="" width="44" height="36" className="h-9 w-auto" />
        <span className="font-display text-sub font-semibold tracking-tight">
          <span className="italic text-ink">Chro</span>
          <span className="text-accent">Lumi</span>
        </span>
      </div>

      {onTab && (
        <nav className="flex flex-1 items-center gap-1" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => onTab(t.id)}
              aria-current={tab === t.id ? "page" : undefined}
              className={`rounded-full px-4 min-h-[40px] text-fine transition-colors duration-200 ${
                t.right ? "ml-auto" : ""
              } ${
                tab === t.id
                  ? "bg-accent text-white font-medium"
                  : "text-ink-mid hover:text-accent hover:bg-white"
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
