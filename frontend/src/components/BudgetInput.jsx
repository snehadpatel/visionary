export default function BudgetInput({ value, onChange }) {
  const presets = [10000, 25000, 50000, 100000]

  return (
    <div className="space-y-4">
      <label className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
        <span className="w-1 h-1 rounded-full bg-ether" />
        Budget Allocation (INR)
      </label>

      <div className="relative group">
        <div className="absolute inset-0 bg-ether/5 blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity" />
        <span className="absolute left-5 top-1/2 -translate-y-1/2 text-muted text-xl font-medium">
          ₹
        </span>
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="e.g. 30,000"
          className="relative w-full bg-surface border border-border-dim rounded-2xl pl-12 pr-6 py-4 text-white text-xl font-bold placeholder-white/10 focus:outline-none focus:border-ether focus:ring-4 focus:ring-ether/5 transition-all"
          id="budget-input"
        />
      </div>

      <div className="flex gap-2 flex-wrap">
        {presets.map((p) => (
          <button
            key={p}
            onClick={() => onChange(String(p))}
            className={`px-4 py-2 rounded-xl text-[11px] font-bold uppercase tracking-wider border transition-all duration-300 cursor-pointer ${
              value === String(p)
                ? "bg-ether border-ether text-white shadow-lg shadow-ether/20"
                : "bg-surface border-border-dim text-muted hover:border-ether/50 hover:text-primary"
            }`}
          >
            ₹{p.toLocaleString("en-IN")}
          </button>
        ))}
      </div>
    </div>
  )
}
