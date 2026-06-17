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
      className="relative w-full max-w-2xl group cursor-pointer"
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
          ? "scale-[1.01]" 
          : "hover:scale-[1.005]"}
      `}
        style={{
          borderColor: dragOver ? 'var(--accent-primary)' : 'rgba(167,139,250,0.3)',
          background: dragOver ? 'rgba(167,139,250,0.06)' : 'rgba(255,255,255,0.5)',
        }}
      >
        {/* Decorative corner elements */}
        <div className="absolute top-6 left-6 w-4 h-4 border-t-2 border-l-2 transition-colors" style={{ borderColor: 'rgba(167,139,250,0.2)' }} />
        <div className="absolute top-6 right-6 w-4 h-4 border-t-2 border-r-2 transition-colors" style={{ borderColor: 'rgba(167,139,250,0.2)' }} />
        <div className="absolute bottom-6 left-6 w-4 h-4 border-b-2 border-l-2 transition-colors" style={{ borderColor: 'rgba(167,139,250,0.2)' }} />
        <div className="absolute bottom-6 right-6 w-4 h-4 border-b-2 border-r-2 transition-colors" style={{ borderColor: 'rgba(167,139,250,0.2)' }} />

        {/* Upload Icon */}
        <div className="relative w-20 h-20 mb-8 flex items-center justify-center">
          <div className="absolute inset-0 rounded-full blur-xl transition-all duration-500" style={{ background: 'rgba(167,139,250,0.12)' }} />
          <div className="relative w-16 h-16 rounded-full glass flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-500"
            style={{ border: '1px solid rgba(167,139,250,0.2)' }}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--accent-primary)' }}>
              <path d="M12 5v14M5 12l7-7 7 7" />
            </svg>
          </div>
        </div>

        <div className="text-center">
          <h3 className="text-xl font-bold mb-2" style={{ color: 'var(--text-heading)' }}>Upload a Property Photo</h3>
          <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Drag your room photo here or <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>browse files</span>
          </p>
        </div>

        <div className="flex gap-4 mt-10">
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl text-[11px] font-medium"
            style={{ background: 'rgba(255,255,255,0.6)', border: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-success)' }} />
            High-Res Support
          </div>
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl text-[11px] font-medium"
            style={{ background: 'rgba(255,255,255,0.6)', border: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-success)' }} />
            AI Room Detection
          </div>
        </div>
      </div>
    </motion.div>
  )
}
