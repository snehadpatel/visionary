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
      <label className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
        Target Style
      </label>

      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
        {STYLES.map((s) => (
          <button
            key={s.id}
            onClick={() => onChange(s.id)}
            className="flex flex-col items-center gap-1.5 p-3 rounded-xl transition-all duration-200 text-center cursor-pointer"
            style={{
              background: value === s.id ? 'rgba(167,139,250,0.1)' : 'var(--bg-card)',
              border: `1px solid ${value === s.id ? 'rgba(167,139,250,0.35)' : 'var(--border-subtle)'}`,
              color: value === s.id ? 'var(--text-heading)' : 'var(--text-secondary)',
              boxShadow: value === s.id ? '0 4px 16px var(--glow-primary)' : '0 1px 3px rgba(0,0,0,0.02)',
            }}
            id={`style-${s.id}`}
          >
            <span className="text-xl">{s.icon}</span>
            <span className="text-xs font-semibold leading-tight">{s.name}</span>
            <span className="text-[10px] leading-tight" style={{ color: 'var(--text-muted)' }}>{s.desc}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
