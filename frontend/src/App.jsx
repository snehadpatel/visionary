import { useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import LandingHero from "./components/LandingHero"
import UploadZone from "./components/UploadZone"
import BudgetInput from "./components/BudgetInput"
import StyleSelector from "./components/StyleSelector"
import PipelineProgress from "./components/PipelineProgress"
import BeforeAfterViewer from "./components/BeforeAfterViewer"
import ShoppingList from "./components/ShoppingList"
import ChatRefinement from "./components/ChatRefinement"
import BudgetAllocationChart from "./components/BudgetAllocationChart"
import CameraPermission from "./components/CameraPermission"
import LiveMode from "./components/LiveMode"
import RoomWalkthrough from "./components/RoomWalkthrough"
import SpatialInsights from "./components/SpatialInsights"
import useVisionaryWS from "./hooks/useVisionaryWS"
import "./index.css"

export default function App() {
  const {
    stage, setStage,
    detections,
    vlmAnalysis,
    budgetAllocation,
    pipelineStep,
    sdPreview,
    matchedProducts,
    result,
    chatTokens,
    messages,
    error,
    uploadImage,
    changeBudget,
    submitRedesign,
    sendChatMessage,
    reset,
    // Live mode
    isLive,
    liveScene,
    liveDetections,
    liveNarration,
    liveNarrationDone,
    quickRedesignPreview,
    startLive,
    stopLive,
    sendLiveFrame,
    sendVoiceInput,
    liveRedesignFrame,
    processHttpFrame,
  } = useVisionaryWS()

  const [localBudget, setLocalBudget] = useState("")
  const [localStyle, setLocalStyle] = useState("auto")
  const [localPrompt, setLocalPrompt] = useState("")
  const [localUseSD, setLocalUseSD] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [showPcd, setShowPcd] = useState(false)

  const handleFileSelect = (file) => {
    setPreviewUrl(URL.createObjectURL(file))
    uploadImage(file)
    setStage("configure")
    setShowPcd(false)
  }

  const handleBudgetUpdate = (val) => {
    setLocalBudget(val)
    changeBudget(parseFloat(val) || 0, localStyle, localPrompt)
  }

  const handleStyleUpdate = (val) => {
    setLocalStyle(val)
    changeBudget(parseFloat(localBudget) || 0, val, localPrompt)
  }

  const handlePromptUpdate = (val) => {
    setLocalPrompt(val)
    changeBudget(parseFloat(localBudget) || 0, localStyle, val)
  }

  return (
    <div className="relative min-h-screen bg-obsidian text-primary overflow-x-hidden">
      {/* Background Layer */}
      <div className="noise-bg" />
      <div className="glow-orb" style={{ top: '20%', left: '10%' }} />
      <div className="glow-orb" style={{ bottom: '10%', right: '10%', opacity: 0.1 }} />

      <AnimatePresence mode="wait">
        {/* LANDING */}
        {stage === "landing" && (
          <LandingHero
            key="landing"
            onStart={() => setStage("upload")}
            onLive={() => setStage("camera-permission")}
          />
        )}

        {/* CAMERA PERMISSION */}
        {stage === "camera-permission" && (
          <CameraPermission
            key="camera-permission"
            onAllow={() => setStage("live")}
            onBack={() => setStage("landing")}
          />
        )}

        {/* LIVE MODE */}
        {stage === "live" && (
          <LiveMode
            key="live"
            liveScene={liveScene}
            liveDetections={liveDetections}
            liveNarration={liveNarration}
            liveNarrationDone={liveNarrationDone}
            liveRedesignFrame={liveRedesignFrame}
            messages={messages}
            isLive={isLive}
            processHttpFrame={processHttpFrame}
            submitRedesign={submitRedesign}
            onExit={() => setStage("landing")}
          />
        )}

        {/* UPLOAD */}
        {stage === "upload" && (
          <motion.div
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative flex flex-col items-center justify-center min-h-screen px-6 py-16"
          >
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="mb-12 text-center"
            >
              <span className="badge badge-active mb-4">Step 01</span>
              <h2 className="mb-2 text-white">Capture Geometry</h2>
              <p className="text-text-secondary">Upload a high-resolution photo of your environment.</p>
            </motion.div>
            <UploadZone onFileSelect={handleFileSelect} />
            <button
              onClick={() => setStage("landing")}
              className="mt-12 text-sm font-bold tracking-widest transition-colors uppercase text-text-muted hover:text-white"
            >
              &larr; Cancel Session
            </button>
          </motion.div>
        )}

        {/* CONFIGURE (The Command Center) */}
        {stage === "configure" && (
          <motion.div
            key="configure"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col min-h-screen"
          >
            {/* Header */}
            <header className="px-8 py-4 border-b glass border-border-subtle flex justify-between items-center z-20">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-accent-primary flex items-center justify-center font-bold text-white shadow-lg shadow-accent-primary/20">V</div>
                <h1 className="text-lg font-bold tracking-tight">Visionary <span className="text-text-muted font-light">Suite</span></h1>
              </div>
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-accent-success" />
                  <span className="text-[10px] font-mono text-text-muted uppercase tracking-widest">Neural Link Active</span>
                </div>
                <button onClick={() => setStage("upload")} className="text-xs font-bold text-text-muted hover:text-white transition-colors">Discard</button>
              </div>
            </header>

            <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-0 overflow-hidden">
              {/* Left: Preview Panel */}
              <div className="lg:col-span-7 bg-black p-8 flex items-center justify-center relative overflow-hidden">
                <div className="glow-orb opacity-10" />
                <motion.div 
                  initial={{ scale: 0.95, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="relative max-w-full max-h-full rounded-2xl overflow-hidden shadow-2xl border border-white/5"
                >
                  <img src={previewUrl} alt="Preview" className="w-auto h-auto max-h-[70vh] object-contain" />
                  
                  {/* Detections Layer */}
                  <AnimatePresence>
                    {detections.map((d, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="absolute border border-accent-primary/40 bg-accent-primary/5 rounded-lg pointer-events-none"
                        style={{
                           left: `${(d.bbox[0] / 1000) * 100}%`,
                           top: `${(d.bbox[1] / 1000) * 100}%`,
                           width: `${((d.bbox[2] - d.bbox[0]) / 1000) * 100}%`,
                           height: `${((d.bbox[3] - d.bbox[1]) / 1000) * 100}%`
                        }}
                      >
                        <span className="absolute -top-5 left-0 px-1.5 py-0.5 bg-accent-primary text-white text-[8px] font-bold uppercase tracking-tighter rounded-sm">
                          {d.label}
                        </span>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </motion.div>

                {/* Status Float */}
                <div className="absolute bottom-8 left-8 flex gap-3">
                  <div className="px-4 py-2 glass rounded-2xl border border-white/5 flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-xs">🧠</div>
                    <div>
                      <p className="text-[10px] font-bold text-text-muted uppercase">Scene Analysis</p>
                      <p className="text-xs font-bold text-white capitalize">{vlmAnalysis?.current_style || 'Detecting...'}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right: Controls Panel */}
              <div className="lg:col-span-5 bg-bg-card border-l border-border-subtle p-10 overflow-y-auto">
                <div className="max-w-md mx-auto space-y-10">
                  <div>
                    <span className="badge badge-active mb-4">Configuration</span>
                    <h2 className="text-2xl font-bold mb-2">Refine Specification</h2>
                    <p className="text-sm text-text-secondary leading-relaxed">Adjust spatial parameters and budget allocation for high-fidelity synthesis.</p>
                  </div>

                  {/* Input Groups */}
                  <section className="space-y-8">
                    <div className="space-y-3">
                      <label className="text-[10px] font-bold uppercase tracking-widest text-text-muted flex items-center gap-2">
                        <span className="w-1 h-1 rounded-full bg-accent-primary" />
                        Design Intent
                      </label>
                      <textarea
                        className="w-full h-24 bg-bg-surface border border-border-subtle rounded-xl p-4 text-sm text-white focus:border-accent-primary focus:ring-1 focus:ring-accent-primary/20 transition-all outline-none resize-none"
                        placeholder="e.g. Minimalist japandi style with walnut textures and warm ambient lighting..."
                        value={localPrompt}
                        onChange={(e) => handlePromptUpdate(e.target.value)}
                      />
                    </div>

                    <BudgetInput value={localBudget} onChange={handleBudgetUpdate} />
                    
                    {budgetAllocation && (
                      <div className="pt-2">
                        <label className="text-[10px] font-bold uppercase tracking-widest text-text-muted mb-4 block">Engine Allocation</label>
                        <BudgetAllocationChart data={budgetAllocation} />
                      </div>
                    )}

                    <div className="space-y-3">
                      <label className="text-[10px] font-bold uppercase tracking-widest text-text-muted flex items-center gap-2">
                        <span className="w-1 h-1 rounded-full bg-accent-primary" />
                        Synthesis Engine
                      </label>
                      <div className="grid grid-cols-2 gap-3">
                        <button
                          onClick={() => setLocalUseSD(false)}
                          className={`py-3 px-4 rounded-xl border text-[10px] font-bold tracking-widest transition-all ${
                            !localUseSD 
                              ? "bg-accent-primary border-accent-primary text-white shadow-lg shadow-accent-primary/20" 
                              : "bg-bg-surface border-border-subtle text-text-muted hover:border-border-bright"
                          }`}
                        >
                          ⚡ FAST (UNet)
                        </button>
                        <button
                          onClick={() => setLocalUseSD(true)}
                          className={`py-3 px-4 rounded-xl border text-[10px] font-bold tracking-widest transition-all ${
                            localUseSD 
                              ? "bg-accent-secondary border-accent-secondary text-white shadow-lg shadow-accent-secondary/20" 
                              : "bg-bg-surface border-border-subtle text-text-muted hover:border-border-bright"
                          }`}
                        >
                          💎 HQ (SD v1.5)
                        </button>
                      </div>
                    </div>
                  </section>

                  {/* Footer Action */}
                  <div className="pt-6 border-t border-border-dim">
                    <button
                      onClick={() => submitRedesign(parseFloat(localBudget), localStyle, localPrompt, localUseSD)}
                      disabled={!localBudget || !vlmAnalysis}
                      className="btn btn-primary w-full h-[60px] text-base rounded-2xl disabled:opacity-30 disabled:grayscale transition-all flex items-center justify-center gap-3"
                    >
                      {!vlmAnalysis ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          Calibrating Neural Link...
                        </>
                      ) : (
                        "Initialize Redesign ✦"
                      )}
                    </button>
                    <p className="mt-4 text-center text-[10px] text-muted">Inference typically takes 8-40 seconds depending on engine.</p>
                  </div>
                </div>
              </div>
            </main>
          </motion.div>
        )}

        {/* PROCESSING */}
        {stage === "processing" && (
          <PipelineProgress
            key="processing"
            step={pipelineStep}
            previewUrl={previewUrl}
            sdPreview={sdPreview}
          />
        )}

        {/* RESULT (The Portfolio View) */}
        {stage === "result" && result && (
          <motion.div
            key="result"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="relative flex flex-col items-center min-h-screen px-6 py-20 overflow-x-hidden"
          >
            {/* Cinematic Header */}
            <div className="mb-20 text-center max-w-2xl">
              <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
                <span className="badge badge-active mb-4">Project Complete</span>
                <h1 className="mb-4 text-white">Spatial Vision</h1>
                <p className="text-text-secondary text-lg font-light leading-relaxed">
                  Redesigned in <span className="text-white font-medium capitalize">{result.target_style}</span> style. 
                  Architectural integrity preserved via ControlNet synthesis.
                </p>
                
                {result.pcd_url && (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setShowPcd(true)}
                    className="mt-8 btn btn-secondary glass mx-auto"
                  >
                    <span className="text-blue-400">🥽</span> Enter 3D Room Walkthrough
                  </motion.button>
                )}
              </motion.div>
            </div>

            {/* Hero Comparison */}
            <section className="w-full mb-32">
              <BeforeAfterViewer
                originalUrl={previewUrl}
                resultUrl={result.result_url}
              />
            </section>

            {/* Spatial Telemetry Dashboard */}
            <SpatialInsights 
              depthUrl={result.depth_url}
              vlmAnalysis={result.vlm_analysis}
              reasoning={result.reasoning}
            />

            {/* Discovery Grid */}
            <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-12 items-start mb-32">
              
              {/* Telemetry Column */}
              <div className="lg:col-span-4 space-y-12">
                <div className="glass-card p-8 border border-border-subtle">
                  <h3 className="text-sm font-bold uppercase tracking-widest mb-6 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-accent-primary" />
                    Spatial Analysis
                  </h3>
                  <div className="space-y-6">
                    {[
                      { label: 'Room Class', value: result.vlm_analysis?.room_type },
                      { label: 'Identified Style', value: result.vlm_analysis?.current_style },
                      { label: 'Geometric Scale', value: result.vlm_analysis?.room_size_estimate },
                      { label: 'Lighting Condition', value: result.vlm_analysis?.natural_light }
                    ].map(item => (
                      <div key={item.label} className="flex justify-between items-end border-b border-white/5 pb-2">
                        <span className="text-[10px] font-mono text-text-muted uppercase">{item.label}</span>
                        <span className="text-xs font-bold text-white capitalize">{item.value || 'N/A'}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <ChatRefinement
                  messages={messages}
                  chatTokens={chatTokens}
                  onSendMessage={sendChatMessage}
                />
              </div>

              {/* Commerce Column */}
              <div className="lg:col-span-8">
                <ShoppingList
                  products={matchedProducts}
                  budgetPlan={result.budget_plan}
                />
              </div>
            </div>

            {/* Reset */}
            <footer className="mt-20 text-center pb-20">
              <button
                onClick={reset}
                className="btn btn-secondary glass h-[54px] px-12 rounded-2xl"
              >
                &larr; Start New Design Project
              </button>
            </footer>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Immersive 3D Room Walkthrough */}
      <AnimatePresence>
        {showPcd && result?.pcd_url && (
          <RoomWalkthrough
            pcdUrl={result.pcd_url}
            resultUrl={result.result_url}
            depthUrl={result.depth_url}
            originalUrl={previewUrl}
            roomInfo={result.vlm_analysis}
            targetStyle={result.target_style}
            objects3D={result.objects_3d}
            onClose={() => setShowPcd(false)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
