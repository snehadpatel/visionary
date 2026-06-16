import { motion } from "framer-motion"

export default function ShoppingList({ products, budgetPlan }) {
  if (!products || products.length === 0) return null

  const total = products.reduce((sum, p) => sum + (p.product?.price_inr || 0), 0)
  const budget = budgetPlan?.total_budget_inr || 0
  const pctUsed = budget > 0 ? Math.round((total / budget) * 100) : 0

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="w-full h-full flex flex-col"
    >
      {/* Commerce Header */}
      <div className="glass-card p-8 border border-border-bright mb-6">
        <div className="flex justify-between items-start mb-8">
          <div>
            <span className="badge badge-active mb-3">Spatial Inventory</span>
            <h3 className="text-white text-xl font-bold tracking-tight">Commerce Mapping</h3>
            <p className="text-text-secondary text-xs font-medium">Real-time matching with Indian e-commerce nodes.</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] font-mono text-text-muted uppercase mb-1">Total Allocation</p>
            <h4 className="text-2xl font-bold text-white">₹{total.toLocaleString("en-IN")}</h4>
          </div>
        </div>

        {/* Budget Progress */}
        <div className="space-y-3">
          <div className="flex justify-between items-end">
            <span className="text-[10px] font-mono text-text-muted uppercase">Utilization Strategy</span>
            <span className={`text-[10px] font-mono font-bold uppercase ${pctUsed > 90 ? 'text-accent-error' : 'text-accent-success'}`}>
              {pctUsed}% Threshold
            </span>
          </div>
          <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(pctUsed, 100)}%` }}
              transition={{ duration: 1, ease: [0.23, 1, 0.32, 1] }}
              className={`h-full rounded-full ${pctUsed > 90 ? 'bg-accent-error' : 'bg-gradient-to-r from-accent-primary to-accent-secondary'}`}
            />
          </div>
          <div className="flex justify-between text-[9px] font-mono text-text-muted">
            <span>₹0</span>
            <span>Target: ₹{budget.toLocaleString("en-IN")}</span>
          </div>
        </div>
      </div>

      {/* Inventory List */}
      <div className="space-y-4">
        {products.map((p, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="flex items-center gap-6 p-4 glass-card bg-bg-surface/40 hover:bg-bg-surface/80 transition-colors group"
          >
            {/* Product Image Node */}
            <div className="relative w-20 h-20 flex-shrink-0">
              {p.product?.image_url ? (
                <img
                  src={p.product.image_url}
                  alt={p.item}
                  className="w-full h-full object-cover rounded-xl border border-white/5 group-hover:border-white/20 transition-colors"
                />
              ) : (
                <div className="w-full h-full rounded-xl bg-black border border-white/5 flex items-center justify-center">
                  <span className="text-2xl grayscale group-hover:grayscale-0 transition-all">🛋️</span>
                </div>
              )}
            </div>

            {/* Content Node */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[9px] font-bold uppercase tracking-widest text-accent-primary">{p.item}</span>
                <span className="w-1 h-1 rounded-full bg-white/10" />
                <span className={`text-[9px] font-bold uppercase tracking-widest ${
                  p.tier === 'premium' ? 'text-accent-secondary' : 'text-text-muted'
                }`}>{p.tier} Tier</span>
              </div>
              <h5 className="text-sm font-bold text-white truncate mb-1">{p.product?.title || "Sourcing product data..."}</h5>
              <p className="text-[10px] text-text-muted font-medium">{p.product?.source || "Global Network"}</p>
            </div>

            {/* Action Node */}
            <div className="text-right flex-shrink-0">
              <p className="text-lg font-bold text-white mb-1">
                {p.product?.price_inr ? `₹${p.product.price_inr.toLocaleString("en-IN")}` : "—"}
              </p>
              {p.product?.url && (
                <motion.a
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  href={p.product.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center text-[10px] font-bold uppercase tracking-widest text-accent-primary hover:text-white transition-colors"
                >
                  Verify Link
                  <svg className="w-3 h-3 ml-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </motion.a>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Analytics Footer */}
      {budgetPlan?.buffer_inr > 0 && (
        <div className="mt-8 p-4 rounded-2xl bg-white/[0.02] border border-white/5 text-center">
          <p className="text-[10px] font-mono text-text-muted uppercase">Contingency Buffer: ₹{budgetPlan.buffer_inr.toLocaleString("en-IN")}</p>
        </div>
      )}
    </motion.div>
  )
}
