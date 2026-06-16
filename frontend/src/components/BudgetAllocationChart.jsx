import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts"
import { motion } from "framer-motion"

const COLORS = ["#6366f1", "#8b5cf6", "#a855f7", "#d946ef", "#ec4899", "#f43f5e"]

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
      className="p-6 rounded-2xl bg-neutral-900 border border-neutral-800 space-y-4"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold uppercase tracking-wider text-neutral-400">Budget Allocation</h3>
        <span className="text-xs font-medium text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full border border-emerald-400/20">
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
                backgroundColor: "#171717",
                border: "1px solid #333",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              itemStyle={{ color: "#fff" }}
              formatter={(value) => `₹${value.toLocaleString("en-IN")}`}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
        {data.allocation.map((item, i) => (
          <div key={item.item} className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
            <span className="text-[11px] text-neutral-400 truncate flex-1">{item.item}</span>
            <span className="text-[11px] text-white font-medium">₹{item.amount_inr.toLocaleString("en-IN")}</span>
          </div>
        ))}
      </div>

      <div className="pt-4 mt-2 border-t border-neutral-800 flex justify-between items-center">
        <span className="text-[10px] text-neutral-500 uppercase font-bold">Total Allocated</span>
        <span className="text-sm font-bold text-white">₹{data.total_allocated.toLocaleString("en-IN")}</span>
      </div>
    </motion.div>
  )
}
