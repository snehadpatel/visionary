import { motion, AnimatePresence } from "framer-motion"

/**
 * Real-time SceneNet analysis displayed as floating glass cards.
 * Shows room type, style, lighting, and color palette.
 */
export default function SceneInfoCards({ scene }) {
  if (!scene) return null

  const roomTypeEmoji = {
    bedroom: "🛏️",
    living_room: "🛋️",
    kitchen: "🍳",
    bathroom: "🚿",
    dining_room: "🍽️",
    office: "💼",
    unknown: "🏠",
  }

  const lightingEmoji = {
    high: "☀️",
    medium: "🌤️",
    low: "🌙",
  }

  return (
    <motion.div
      initial={{ x: 20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      className="flex flex-col gap-2.5 w-52"
    >
      {/* Room Type */}
      <div className="live-info-card">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold">
            Room Type
          </span>
          <span className="text-base">
            {roomTypeEmoji[scene.room_type] || "🏠"}
          </span>
        </div>
        <p className="text-sm font-semibold text-white capitalize">
          {(scene.room_type || "unknown").replace("_", " ")}
        </p>
        <ConfidenceBar
          value={scene.room_type_confidence}
          color="indigo"
        />
      </div>

      {/* Style */}
      <div className="live-info-card">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold">
            Style
          </span>
          <span className="text-base">🎨</span>
        </div>
        <p className="text-sm font-semibold text-white capitalize">
          {(scene.style || "unknown").replace("_", " ")}
        </p>
        <ConfidenceBar
          value={scene.style_confidence}
          color="purple"
        />
        {/* Top 3 style probabilities */}
        {scene.style_probs && (
          <div className="mt-2 space-y-1">
            {Object.entries(scene.style_probs)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 3)
              .map(([style, prob]) => (
                <div key={style} className="flex items-center gap-2 text-[10px]">
                  <span className="text-neutral-500 w-16 capitalize truncate">
                    {style.replace("_", " ")}
                  </span>
                  <div className="flex-1 h-1 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full bg-purple-500/60"
                      initial={{ width: 0 }}
                      animate={{ width: `${prob * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  <span className="text-neutral-600 w-8 text-right">
                    {Math.round(prob * 100)}%
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>

      {/* Lighting */}
      <div className="live-info-card">
        <div className="flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold">
            Lighting
          </span>
          <span className="text-base">
            {lightingEmoji[scene.lighting] || "🌤️"}
          </span>
        </div>
        <p className="text-sm font-semibold text-white capitalize mt-1">
          {scene.lighting || "medium"}
        </p>
      </div>

      {/* Color Palette */}
      {scene.palette && (
        <div className="live-info-card">
          <span className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold block mb-2">
            Palette
          </span>
          <div className="flex gap-1.5">
            {scene.palette.map((color, i) => (
              <motion.div
                key={i}
                className="flex-1 h-6 rounded-lg border border-white/10"
                style={{
                  backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
                }}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: i * 0.1 }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Inference Speed */}
      {scene.last_scenenet_ms > 0 && (
        <div className="text-[10px] text-neutral-600 text-center mt-1">
          SceneNet: {scene.last_scenenet_ms.toFixed(0)}ms
          {scene.last_yolo_ms > 0 && ` · YOLO: ${scene.last_yolo_ms.toFixed(0)}ms`}
        </div>
      )}
    </motion.div>
  )
}

function ConfidenceBar({ value = 0, color = "indigo" }) {
  const colors = {
    indigo: "from-indigo-500 to-indigo-400",
    purple: "from-purple-500 to-purple-400",
    emerald: "from-emerald-500 to-emerald-400",
  }
  return (
    <div className="h-1 bg-white/5 rounded-full overflow-hidden mt-1.5">
      <motion.div
        className={`h-full rounded-full bg-gradient-to-r ${colors[color]}`}
        initial={{ width: 0 }}
        animate={{ width: `${Math.min(value * 100, 100)}%` }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      />
    </div>
  )
}
