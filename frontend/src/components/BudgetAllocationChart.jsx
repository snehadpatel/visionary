import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts"
import { motion } from "framer-motion"

const COLORS = ["#a78bfa", "#f9a8d4", "#6ee7b7", "#fbbf24", "#7dd3fc", "#c4b5fd"]

export default function BudgetAllocationChart({ data }) {
  if (!data || !data.allocation) return null

  const chartData = data.allocation.map((item) => ({
    name: item.item,
    value: item.amount_inr,
  }))

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="p-6 rounded-2xl space-y-4"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', boxShadow: '0 4px 24px rgba(0,0,0,0.04)' }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>Budget Allocation</h3>
        <span className="text-xs font-medium px-2.5 py-0.5 rounded-full" 
          style={{ color: '#059669', background: 'rgba(110,231,183,0.12)', border: '1px solid rgba(110,231,183,0.25)' }}
        >
          Live Preview
        </span>
      </div>

      <div className="h-[200px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
              animationBegin={0}
              animationDuration={800}
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: "#ffffff",
                border: "1px solid rgba(0,0,0,0.08)",
                borderRadius: "12px",
                fontSize: "12px",
                boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
              }}
              itemStyle={{ color: "var(--text-heading)" }}
              formatter={(value) => `₹${value.toLocaleString("en-IN")}`}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
        {data.allocation.map((item, i) => (
          <div key={item.item} className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
            <span className="text-[11px] truncate flex-1" style={{ color: 'var(--text-secondary)' }}>{item.item}</span>
            <span className="text-[11px] font-medium" style={{ color: 'var(--text-heading)' }}>₹{item.amount_inr.toLocaleString("en-IN")}</span>
          </div>
        ))}
      </div>

      <div className="pt-4 mt-2 flex justify-between items-center" style={{ borderTop: '1px solid var(--border-subtle)' }}>
        <span className="text-[10px] uppercase font-semibold" style={{ color: 'var(--text-muted)' }}>Total Allocated</span>
        <span className="text-sm font-bold" style={{ color: 'var(--text-heading)' }}>₹{data.total_allocated.toLocaleString("en-IN")}</span>
      </div>
    </motion.div>
  )
}
