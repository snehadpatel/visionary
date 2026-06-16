import { motion } from "framer-motion"

export default function LandingHero({ onStart, onLive }) {
  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen px-6 overflow-hidden text-center">
      {/* Background Cinematic Orbs */}
      <div className="glow-orb" style={{ top: '10%', left: '10%' }} />
      <div className="glow-orb" style={{ bottom: '10%', right: '10%', background: 'radial-gradient(circle, rgba(168, 85, 247, 0.08) 0%, transparent 70%)' }} />

      <motion.div
        initial={{ y: 40, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2, duration: 1, ease: [0.23, 1, 0.32, 1] }}
        className="relative z-10 max-w-4xl"
      >
        {/* Technical Badge */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-white/5 bg-white/5 backdrop-blur-md text-[11px] font-bold uppercase tracking-[0.2em] text-secondary mb-10"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-success shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
          Native Spatial Engine &middot; Local Inference &middot; Zero Cloud
        </motion.div>

        {/* Hero Title */}
        <h1 className="mb-8 overflow-hidden text-white">
          <motion.span 
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            transition={{ delay: 0.3, duration: 0.8, ease: [0.23, 1, 0.32, 1] }}
            className="block"
          >
            Spatial Design
          </motion.span>
          <motion.span 
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            transition={{ delay: 0.4, duration: 0.8, ease: [0.23, 1, 0.32, 1] }}
            className="block gradient-text"
          >
            Redefined.
          </motion.span>
        </h1>

        <p className="max-w-2xl mx-auto mb-12 text-lg font-light leading-relaxed text-secondary">
          Experience the future of interior architecture. Visionary leverages custom-trained 
          <span className="text-white font-medium"> SceneNet telemetry</span> and 
          <span className="text-white font-medium"> spatial VLM</span> to transform your space with architectural precision.
        </p>

        {/* Actions */}
        <div className="flex flex-col items-center justify-center gap-6 sm:flex-row">
          <motion.button
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={onStart}
            className="btn btn-primary h-[60px] px-12 text-base rounded-2xl w-full sm:w-auto"
            id="start-redesign-btn"
          >
            Start Project
            <svg className="w-5 h-5 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={onLive}
            className="btn btn-secondary h-[60px] px-10 text-base rounded-2xl w-full sm:w-auto glass"
            id="start-live-btn"
          >
            <span className="flex items-center gap-3">
              <span className="relative flex w-2.5 h-2.5">
                <span className="absolute inline-flex w-full h-full rounded-full opacity-75 animate-ping bg-error"></span>
                <span className="relative inline-flex w-2.5 h-2.5 rounded-full bg-error"></span>
              </span>
              Live Spatial Mode
            </span>
          </motion.button>
        </div>

        {/* Tech Stack Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 1 }}
          className="flex flex-wrap justify-center gap-3 mt-20"
        >
          {["YOLOv8x", "SAM-H", "MiDaS DPT", "SD v1.5", "TinyLlama", "SceneNet"].map(tech => (
            <span key={tech} className="px-4 py-1.5 rounded-full border border-white/5 bg-white/[0.02] text-[10px] font-mono text-muted hover:border-white/10 hover:text-secondary transition-colors">
              {tech}
            </span>
          ))}
        </motion.div>
      </motion.div>

      <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
    </div>
  )
}
