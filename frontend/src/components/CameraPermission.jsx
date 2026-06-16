import { motion } from "framer-motion"

/**
 * Dedicated camera permission screen.
 * Shows before accessing camera with explanation of what it's used for.
 */
export default function CameraPermission({ onAllow, onBack }) {
  const isSecure = window.isSecureContext || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="min-h-screen flex flex-col items-center justify-center px-6 relative"
    >
      {!isSecure && (
        <motion.div 
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="absolute top-10 left-6 right-6 p-4 rounded-2xl bg-red-500/10 border border-red-500/30 text-red-400 text-center z-50 backdrop-blur-xl"
        >
          <p className="text-sm font-bold mb-1">⚠️ Insecure Context Detected</p>
          <p className="text-[11px] leading-relaxed opacity-80">
            Mobile browsers block camera access on non-HTTPS sites. 
            Please use <b>HTTPS</b> or <b>localhost</b> port forwarding to test on mobile.
          </p>
        </motion.div>
      )}
      {/* Ambient glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full bg-indigo-500/5 blur-[120px] pointer-events-none" />

      <motion.div
        initial={{ y: 30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="relative z-10 text-center max-w-md"
      >
        {/* Camera Icon */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", delay: 0.2, stiffness: 200 }}
          className="w-24 h-24 rounded-3xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/30 flex items-center justify-center mx-auto mb-8"
        >
          <span className="text-5xl">📷</span>
        </motion.div>

        <h2 className="text-3xl font-bold mb-3">Camera Access Needed</h2>
        <p className="text-neutral-400 mb-8 leading-relaxed">
          Visionary Live uses your camera to analyze rooms in{" "}
          <span className="text-indigo-400 font-medium">real-time</span>. Our AI
          will detect furniture, identify styles, and suggest redesigns as you
          point your camera around.
        </p>

        {/* What we do with camera */}
        <div className="space-y-3 text-left mb-10">
          {[
            {
              icon: "🔍",
              title: "Detect Objects",
              desc: "AI identifies furniture and decor in real-time",
            },
            {
              icon: "🎨",
              title: "Analyze Style",
              desc: "Custom SceneNet classifies room style instantly",
            },
            {
              icon: "🗣️",
              title: "Voice Chat",
              desc: "Talk to AI about what you see for redesign ideas",
            },
          ].map((item) => (
            <div
              key={item.title}
              className="flex items-start gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/5"
            >
              <span className="text-xl mt-0.5">{item.icon}</span>
              <div>
                <p className="text-sm font-semibold text-white">{item.title}</p>
                <p className="text-xs text-neutral-500">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Debug Info (Helpful for mobile) */}
        <div className="mt-4 p-3 rounded-xl bg-black/40 border border-white/5 text-[10px] font-mono text-left space-y-1">
          <p className="text-neutral-500 uppercase font-bold mb-1">System Diagnostics</p>
          <div className="flex justify-between">
            <span>Secure Context:</span>
            <span className={isSecure ? "text-green-500" : "text-red-500"}>{isSecure ? "YES" : "NO"}</span>
          </div>
          <div className="flex justify-between">
            <span>MediaDevices Support:</span>
            <span className={!!navigator.mediaDevices ? "text-green-500" : "text-red-500"}>{!!navigator.mediaDevices ? "YES" : "NO"}</span>
          </div>
          <div className="flex justify-between">
            <span>Protocol:</span>
            <span className="text-indigo-400">{window.location.protocol}</span>
          </div>
          <div className="flex justify-between">
            <span>Hostname:</span>
            <span className="text-indigo-400">{window.location.hostname}</span>
          </div>
        </div>

        {/* Privacy note */}
        <div className="flex items-center gap-2 justify-center my-6 text-xs text-neutral-600">
          <span>🔒</span>
          <span>
            Camera feed is processed locally. Nothing is stored.
          </span>
        </div>

        {/* Actions */}
        <motion.button
          whileHover={{ scale: 1.03, y: -2 }}
          whileTap={{ scale: 0.97 }}
          onClick={onAllow}
          disabled={!isSecure && window.location.hostname !== "localhost"}
          className={`btn-primary w-full py-4 text-lg rounded-2xl glow-primary mb-3 ${!isSecure ? "opacity-50 grayscale" : ""}`}
        >
          {isSecure ? "Allow Camera Access" : "HTTPS Required for Mobile"}
        </motion.button>

        <button
          onClick={onBack}
          className="w-full text-center text-sm text-neutral-600 hover:text-neutral-400 transition-colors cursor-pointer py-2"
        >
          &larr; Go back
        </button>
      </motion.div>
    </motion.div>
  )
}
