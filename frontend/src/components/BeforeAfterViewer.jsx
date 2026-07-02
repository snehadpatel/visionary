import { useState, useRef, useEffect } from "react"
import { motion } from "framer-motion"

export default function BeforeAfterViewer({ originalUrl, resultUrl }) {
  const [sliderPosition, setSliderPosition] = useState(50)
  const containerRef = useRef(null)
  const isDragging = useRef(false)

  const handleMove = (e) => {
    if (!isDragging.current || !containerRef.current) return
    
    const rect = containerRef.current.getBoundingClientRect()
    const x = e.type.includes('touch') ? e.touches[0].clientX : e.clientX
    const position = ((x - rect.left) / rect.width) * 100
    
    setSliderPosition(Math.min(Math.max(position, 0.1), 99.9))
  }

  const handleStart = () => (isDragging.current = true)
  const handleEnd = () => (isDragging.current = false)

  useEffect(() => {
    window.addEventListener('mouseup', handleEnd)
    window.addEventListener('touchend', handleEnd)
    window.addEventListener('mousemove', handleMove)
    window.addEventListener('touchmove', handleMove)
    return () => {
      window.removeEventListener('mouseup', handleEnd)
      window.removeEventListener('touchend', handleEnd)
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('touchmove', handleMove)
    }
  }, [])

  return (
    <div className="w-full max-w-5xl px-4 mx-auto">
      <div 
        ref={containerRef}
        className="relative aspect-video rounded-[32px] overflow-hidden glass-card border border-[var(--border-bright)] shadow-2xl cursor-ew-resize select-none"
        onMouseDown={handleStart}
        onTouchStart={handleStart}
      >
        {/* Result Image (Bottom Layer) */}
        <img 
          src={resultUrl} 
          className="absolute inset-0 object-cover w-full h-full"
          alt="Redesign"
        />

        {/* Original Image (Top Layer, Clipped) */}
        <div 
          className="absolute inset-0 z-10 overflow-hidden"
          style={{ width: `${sliderPosition}%` }}
        >
          <img 
            src={originalUrl} 
            className="absolute inset-0 object-cover h-full"
            style={{ width: `${100 / (sliderPosition / 100)}%` }}
            alt="Original"
          />
        </div>

        {/* Slider Handle */}
        <div 
          className="absolute inset-y-0 z-20 w-px bg-white/50 backdrop-blur-md"
          style={{ left: `${sliderPosition}%` }}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 rounded-full glass border border-white/20 flex items-center justify-center shadow-2xl">
            <div className="flex gap-1">
              <div className="w-0.5 h-3 bg-white/60 rounded-full" />
              <div className="w-0.5 h-3 bg-white/60 rounded-full" />
            </div>
          </div>
        </div>

        {/* Labels */}
        <div className="absolute top-6 left-6 z-30 px-3 py-1 glass rounded-full text-[10px] font-bold text-[var(--text-secondary)] uppercase tracking-widest border border-[var(--border-subtle)] pointer-events-none">
          Before
        </div>
        <div className="absolute top-6 right-6 z-30 px-3 py-1 glass rounded-full text-[10px] font-bold text-[var(--accent-primary)] uppercase tracking-widest border border-[var(--accent-primary)]/20 pointer-events-none">
          After
        </div>
      </div>

      <div className="flex flex-col items-center justify-between gap-6 mt-8 sm:flex-row">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-2xl bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
            <svg className="w-5 h-5 text-[var(--accent-primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-bold text-[var(--text-heading)]">Before & After comparison</h4>
            <p className="text-xs text-[var(--text-muted)]">Slide to compare architectural changes</p>
          </div>
        </div>

        <motion.a
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          href={resultUrl}
          download="visionary-redesign.png"
          className="btn btn-secondary group"
        >
          <svg className="w-4 h-4 mr-2 transition-transform group-hover:translate-y-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1h16v-1M12 12V4m0 8l-4-4m4 4l4-4" />
          </svg>
          Download Redesign
        </motion.a>
      </div>
    </div>
  )
}
