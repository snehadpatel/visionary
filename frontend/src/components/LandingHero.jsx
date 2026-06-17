import { motion } from "framer-motion"

export default function LandingHero({ onStart, onLive }) {
  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen px-6 overflow-hidden text-center">
      {/* Background Pastel Orbs */}
      <div className="glow-orb" style={{ top: '10%', left: '5%' }} />
      <div className="glow-orb" style={{ bottom: '10%', right: '5%', background: 'radial-gradient(circle, rgba(249, 168, 212, 0.15) 0%, transparent 70%)' }} />

      <motion.div
        initial={{ y: 40, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2, duration: 1, ease: [0.23, 1, 0.32, 1] }}
        className="relative z-10 max-w-4xl"
      >
        {/* Proptech Badge */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="inline-flex items-center gap-2 px-5 py-2 rounded-full border bg-white/60 backdrop-blur-md text-[12px] font-semibold tracking-wide mb-10"
          style={{ borderColor: 'rgba(0,0,0,0.08)', color: 'var(--text-secondary)' }}
        >
          <span className="w-2 h-2 rounded-full shadow-sm" style={{ background: 'var(--accent-success)', boxShadow: '0 0 8px rgba(110,231,183,0.5)' }} />
          AI Virtual Staging &middot; Smart Budgeting &middot; Instant Redesign
        </motion.div>

        {/* Hero Title */}
        <h1 className="mb-8 overflow-hidden" style={{ fontSize: 'clamp(2.5rem, 6vw, 4.5rem)', lineHeight: 1.1 }}>
          <motion.span 
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            transition={{ delay: 0.3, duration: 0.8, ease: [0.23, 1, 0.32, 1] }}
            className="block"
            style={{ color: 'var(--text-heading)' }}
          >
            Redesign Any Property,
          </motion.span>
          <motion.span 
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            transition={{ delay: 0.4, duration: 0.8, ease: [0.23, 1, 0.32, 1] }}
            className="block gradient-text"
          >
            Instantly.
          </motion.span>
        </h1>

        <p className="max-w-2xl mx-auto mb-12 text-lg font-light leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          AI-powered interior redesign for real estate professionals and homeowners. 
          Upload a photo, pick a style, and get a{' '}
          <span style={{ color: 'var(--text-heading)', fontWeight: 500 }}>stunning virtual staging</span> with{' '}
          <span style={{ color: 'var(--text-heading)', fontWeight: 500 }}>shoppable furniture picks</span> — in seconds.
        </p>

        {/* Actions */}
        <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
          <motion.button
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={onStart}
            className="btn btn-primary h-[56px] px-10 text-[15px] rounded-2xl w-full sm:w-auto"
            id="start-redesign-btn"
          >
            Start Redesign
            <svg className="w-5 h-5 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={onLive}
            className="btn btn-secondary h-[56px] px-8 text-[15px] rounded-2xl w-full sm:w-auto"
            id="start-live-btn"
          >
            <span className="flex items-center gap-3">
              <span className="relative flex w-2.5 h-2.5">
                <span className="absolute inline-flex w-full h-full rounded-full opacity-75 animate-ping" style={{ background: 'var(--accent-error)' }}></span>
                <span className="relative inline-flex w-2.5 h-2.5 rounded-full" style={{ background: 'var(--accent-error)' }}></span>
              </span>
              Live Property Scan
            </span>
          </motion.button>
        </div>

        {/* Proptech Feature Pills */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 1 }}
          className="flex flex-wrap justify-center gap-3 mt-16"
        >
          {["Virtual Staging", "Smart Layout", "Depth Mapping", "AI Render", "Style Match", "3D Walkthrough"].map(tech => (
            <span key={tech} className="px-4 py-1.5 rounded-full border text-[11px] font-medium transition-colors cursor-default"
              style={{ 
                borderColor: 'var(--border-subtle)', 
                background: 'rgba(255,255,255,0.5)',
                color: 'var(--text-secondary)'
              }}
            >
              {tech}
            </span>
          ))}
        </motion.div>
      </motion.div>

      <div className="absolute bottom-0 left-0 w-full h-px" style={{ background: 'linear-gradient(to right, transparent, var(--border-bright), transparent)' }} />
    </div>
  )
}
