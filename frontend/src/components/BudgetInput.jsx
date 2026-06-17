export default function BudgetInput({ value, onChange }) {
  const presets = [10000, 25000, 50000, 100000]

  return (
    <div className="space-y-4">
      <label className="text-[11px] font-semibold uppercase tracking-wide flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-primary)' }} />
        Budget Allocation (INR)
      </label>

      <div className="relative group">
        <span className="absolute left-5 top-1/2 -translate-y-1/2 text-xl font-medium" style={{ color: 'var(--text-muted)' }}>
          ₹
        </span>
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="e.g. 30,000"
          className="relative w-full rounded-2xl pl-12 pr-6 py-4 text-xl font-bold transition-all outline-none"
          style={{ 
            background: 'var(--bg-surface)', 
            border: '1px solid var(--border-subtle)', 
            color: 'var(--text-heading)',
          }}
          onFocus={(e) => { e.target.style.borderColor = 'var(--accent-primary)'; e.target.style.boxShadow = '0 0 0 3px var(--glow-primary)'; }}
          onBlur={(e) => { e.target.style.borderColor = 'var(--border-subtle)'; e.target.style.boxShadow = 'none'; }}
          id="budget-input"
        />
      </div>

      <div className="flex gap-2 flex-wrap">
        {presets.map((p) => (
          <button
            key={p}
            onClick={() => onChange(String(p))}
            className="px-4 py-2 rounded-xl text-[11px] font-semibold tracking-wide transition-all duration-300 cursor-pointer"
            style={{
              background: value === String(p) ? 'rgba(167,139,250,0.12)' : 'var(--bg-surface)',
              border: `1px solid ${value === String(p) ? 'rgba(167,139,250,0.4)' : 'var(--border-subtle)'}`,
              color: value === String(p) ? 'var(--accent-primary)' : 'var(--text-muted)',
              boxShadow: value === String(p) ? '0 4px 12px var(--glow-primary)' : 'none',
            }}
          >
            ₹{p.toLocaleString("en-IN")}
          </button>
        ))}
      </div>
    </div>
  )
}
