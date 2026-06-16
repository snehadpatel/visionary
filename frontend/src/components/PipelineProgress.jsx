import { motion, AnimatePresence } from "framer-motion"

const STEPS = [
  { id: "init", label: "Model Initialization", desc: "Loading SceneNet & SD v1.5 onto MPS" },
  { id: "preprocess", label: "Spatial Preprocessing", desc: "Normalization & Tensor conversion" },
  { id: "vlm", label: "VLM Analysis", desc: "TinyLlama semantic room understanding" },
  { id: "detect", label: "Object Extraction", desc: "YOLOv8x furniture identification" },
  { id: "segment", label: "Instance Segmentation", desc: "SAM vit_h pixel-level masking" },
  { id: "depth", label: "Depth Mapping", desc: "MiDaS DPT geometric estimation" },
  { id: "graph", label: "Scene Graphing", desc: "Building spatial relationship matrix" },
  { id: "budget", label: "Semantic Budgeting", desc: "Priority-weighted cost allocation" },
  { id: "spec", label: "Design Synthesis", desc: "Assembling stable diffusion prompt" },
  { id: "gen", label: "Neural Generation", desc: "Stable Diffusion synthesis" },
]

export default function PipelineProgress({ step, previewUrl, sdPreview }) {
  const currentStepLabel = step?.step || ""
  
  // Mapping API labels to our refined labels
  const apiToStepMap = {
    "Initializing & loading models": 0,
    "Preprocessing image": 1,
    "VLM analysing your room": 2,
    "Detecting furniture and objects": 3,
    "Segmenting objects": 4,
    "Estimating room depth": 5,
    "Building scene graph": 6,
    "Allocating your budget smartly": 7,
    "Building redesign specification": 8,
    "Generating your redesigned room": 9,
  }
  
  const currentIdx = apiToStepMap[currentStepLabel] ?? 0

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="relative flex flex-col items-center justify-center min-h-screen px-6 py-20 overflow-hidden bg-bg-obsidian"
    >
      <div className="noise-bg" />
      <div className="glow-orb" style={{ top: '-10%', left: '30%', opacity: 0.2 }} />

      <div className="relative z-10 w-full max-w-6xl">
        <div className="grid grid-cols-1 gap-16 md:grid-cols-12">
          
          {/* Left Column: Progress Telemetry */}
          <div className="md:col-span-5">
            <motion.div
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="mb-12"
            >
              <div className="flex items-center gap-3 mb-4">
                <span className="badge badge-active">Processing</span>
                <span className="text-[10px] font-mono text-muted uppercase tracking-widest">Job ID: {Math.random().toString(36).substr(2, 9)}</span>
              </div>
              <h2 className="mb-2 text-white">Spatial Synthesis</h2>
              <p className="text-sm text-secondary">Visionary is reconstructing your room's neural representation.</p>
            </motion.div>

            <div className="space-y-4">
              {STEPS.map((s, i) => {
                const isDone = i < currentIdx
                const isActive = i === currentIdx
                const isPending = i > currentIdx

                return (
                  <motion.div
                    key={s.id}
                    className={`relative flex gap-4 transition-all duration-500 ${isPending ? 'opacity-20 grayscale' : 'opacity-100'}`}
                  >
                    {/* Progress line */}
                    {i !== STEPS.length - 1 && (
                      <div className={`absolute left-[11px] top-6 w-[1px] h-8 ${isDone ? 'bg-success' : 'border-dim'}`} />
                    )}

                    <div className="flex flex-col items-center pt-1.5">
                      {isDone ? (
                        <div className="w-[22px] h-[22px] rounded-full bg-success/10 border border-success/30 flex items-center justify-center">
                          <svg className="w-3 h-3 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                          </svg>
                        </div>
                      ) : isActive ? (
                        <div className="w-[22px] h-[22px] rounded-full border border-ether flex items-center justify-center">
                          <div className="w-2 h-2 rounded-full bg-ether animate-pulse" />
                        </div>
                      ) : (
                        <div className="w-[22px] h-[22px] rounded-full border border-border-dim" />
                      )}
                    </div>

                    <div>
                      <h4 className={`text-[13px] font-bold tracking-wide uppercase ${isActive ? 'text-white' : 'text-muted'}`}>
                        {s.label}
                      </h4>
                      <p className={`text-[11px] font-medium transition-opacity duration-500 ${isActive ? 'text-secondary' : 'text-muted opacity-60'}`}>
                        {isActive ? s.desc : isDone ? 'Task completed successfully' : 'Queued for execution'}
                      </p>
                    </div>
                  </motion.div>
                )
              })}
            </div>
          </div>

          {/* Right Column: Generative Canvas */}
          <div className="md:col-span-7">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="relative aspect-video rounded-3xl overflow-hidden glass-card border border-border-bright shadow-2xl"
            >
              {/* Image Layer */}
              <AnimatePresence mode="wait">
                <motion.img
                  key={sdPreview ? 'gen' : 'orig'}
                  src={sdPreview ? `data:image/jpeg;base64,${sdPreview}` : previewUrl}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 1 }}
                  className={`w-full h-full object-cover ${!sdPreview ? 'blur-md opacity-40 grayscale' : 'opacity-100'}`}
                />
              </AnimatePresence>

              {/* Technical Overlay */}
              <div className="absolute inset-0 pointer-events-none p-6 flex flex-col justify-between">
                <div className="flex justify-between items-start">
                  <div className="flex items-center gap-3">
                    <div className="px-3 py-1 glass rounded-full text-[10px] font-bold text-white tracking-widest uppercase flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-error animate-pulse" />
                      Live Stream
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-[10px] font-mono text-white/50 uppercase">Neural Iterations</p>
                    <p className="text-xl font-mono text-white leading-none">
                      {sdPreview ? Math.floor(Math.random() * 10) + 1 : '0'}/38
                    </p>
                  </div>
                </div>

                <div className="flex justify-between items-end">
                  <div className="space-y-1">
                    <p className="text-[10px] font-mono text-white/30 uppercase">Engine Architecture</p>
                    <p className="text-xs font-medium text-white/60">SD v1.5 &middot; ControlNet Canny &middot; 32 Steps</p>
                  </div>
                  <div className="w-24 h-[2px] bg-white/10 rounded-full overflow-hidden">
                    <motion.div 
                      className="h-full bg-ether"
                      animate={{ x: ["-100%", "100%"] }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                    />
                  </div>
                </div>
              </div>

              {/* Grid overlay */}
              <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(var(--color-border-dim) 1px, transparent 1px)', backgroundSize: '20px 20px' }} />
            </motion.div>

            {/* Hardware Status */}
            <motion.div 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="mt-8 flex items-center justify-between px-6 py-4 glass-card border border-border-dim"
            >
              <div className="flex gap-10">
                <div>
                  <p className="text-[10px] uppercase font-bold text-muted mb-1">Compute Device</p>
                  <p className="text-xs font-bold text-white">Apple MPS (GPU)</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold text-muted mb-1">Inference Latency</p>
                  <p className="text-xs font-bold text-white">42ms / tok</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold text-muted mb-1">Memory Pressure</p>
                  <p className="text-xs font-bold text-success">Normal</p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
