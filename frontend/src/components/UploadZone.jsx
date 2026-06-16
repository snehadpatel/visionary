import { useState, useRef } from "react"
import { motion } from "framer-motion"

export default function UploadZone({ onFileSelect }) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith("image/")) {
      onFileSelect(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)
  const handleClick = () => inputRef.current?.click()

  const handleChange = (e) => {
    const file = e.target.files[0]
    if (file) onFileSelect(file)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative w-full max-w-2xl group cursor-pointer`}
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      id="upload-zone"
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden"
        id="file-input"
      />

      <div className={`
        relative flex flex-col items-center justify-center h-[340px] rounded-[32px] border-2 border-dashed transition-all duration-500
        ${dragOver 
          ? "border-accent-primary bg-accent-primary/5 scale-[1.01]" 
          : "border-border-subtle bg-white/[0.02] hover:border-border-bright hover:bg-white/[0.04]"}
      `}>
        {/* Decorative corner elements */}
        <div className="absolute top-6 left-6 w-4 h-4 border-t-2 border-l-2 border-white/10 group-hover:border-white/20 transition-colors" />
        <div className="absolute top-6 right-6 w-4 h-4 border-t-2 border-r-2 border-white/10 group-hover:border-white/20 transition-colors" />
        <div className="absolute bottom-6 left-6 w-4 h-4 border-b-2 border-l-2 border-white/10 group-hover:border-white/20 transition-colors" />
        <div className="absolute bottom-6 right-6 w-4 h-4 border-b-2 border-r-2 border-white/10 group-hover:border-white/20 transition-colors" />

        {/* Upload Icon Sphere */}
        <div className="relative w-20 h-20 mb-8 flex items-center justify-center">
          <div className="absolute inset-0 rounded-full bg-accent-primary/10 blur-xl group-hover:bg-accent-primary/20 transition-all duration-500" />
          <div className="relative w-16 h-16 rounded-full glass border border-white/10 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-500">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white">
              <path d="M12 5v14M5 12l7-7 7 7" />
            </svg>
          </div>
        </div>

        <div className="text-center">
          <h3 className="text-xl font-bold text-white mb-2 tracking-tight">Import Space Geometry</h3>
          <p className="text-text-secondary text-sm font-medium">Drag room capture or <span className="text-accent-primary">browse files</span></p>
        </div>

        <div className="flex gap-4 mt-10">
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-black/40 border border-white/5 text-[10px] font-mono text-text-muted uppercase tracking-widest">
            <span className="w-1.5 h-1.5 rounded-full bg-white/20" />
            8K Resolution Support
          </div>
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-black/40 border border-white/5 text-[10px] font-mono text-text-muted uppercase tracking-widest">
            <span className="w-1.5 h-1.5 rounded-full bg-white/20" />
            Auto-Spatial Scaling
          </div>
        </div>
      </div>
    </motion.div>
  )
}
