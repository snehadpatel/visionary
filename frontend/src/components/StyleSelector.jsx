const STYLES = [
  { id: "auto", name: "Auto-Detect", icon: "✦", desc: "Let AI choose" },
  { id: "scandinavian", name: "Scandinavian", icon: "🪵", desc: "Light, minimal, hygge" },
  { id: "industrial", name: "Industrial", icon: "⚙️", desc: "Brick, steel, concrete" },
  { id: "bohemian", name: "Bohemian", icon: "🌿", desc: "Rattan, textiles, warm" },
  { id: "mid-century modern", name: "Mid-Century", icon: "🪑", desc: "Teak, retro, tapered" },
  { id: "japandi", name: "Japandi", icon: "🎋", desc: "Bamboo, wabi-sabi" },
  { id: "minimalist", name: "Minimalist", icon: "◻️", desc: "Clean, pure, space" },
  { id: "luxury", name: "Luxury", icon: "✨", desc: "Marble, velvet, gold" },
  { id: "coastal", name: "Coastal", icon: "🐚", desc: "Beach, white, blue" },
  { id: "rustic", name: "Rustic", icon: "🏡", desc: "Wood, stone, plaid" },
]

export default function StyleSelector({ value, onChange }) {
  return (
    <div className="space-y-3">
      <label className="text-xs font-bold uppercase tracking-wider text-neutral-500">
        Target Style
      </label>

      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
        {STYLES.map((s) => (
          <button
            key={s.id}
            onClick={() => onChange(s.id)}
            className={`flex flex-col items-center gap-1.5 p-3 rounded-xl border transition-all duration-200 text-center cursor-pointer ${
              value === s.id
                ? "bg-indigo-500/10 border-indigo-500/40 text-white shadow-lg shadow-indigo-500/10"
                : "border-neutral-800 text-neutral-400 hover:border-neutral-600 hover:text-neutral-300 bg-neutral-900/50"
            }`}
            id={`style-${s.id}`}
          >
            <span className="text-xl">{s.icon}</span>
            <span className="text-xs font-semibold leading-tight">{s.name}</span>
            <span className="text-[10px] text-neutral-500 leading-tight">{s.desc}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
