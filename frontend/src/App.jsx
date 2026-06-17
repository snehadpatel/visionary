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
    <div className="relative min-h-screen overflow-x-hidden" style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      {/* Background Layer */}
      <div className="noise-bg" />
      <div className="glow-orb" style={{ top: '20%', left: '10%' }} />
      <div className="glow-orb" style={{ bottom: '10%', right: '10%', opacity: 0.08, background: 'radial-gradient(circle, var(--accent-secondary) 0%, transparent 70%)' }} />

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
              <h2 className="mb-2" style={{ color: 'var(--text-heading)' }}>Scan Your Space</h2>
              <p style={{ color: 'var(--text-secondary)' }}>Upload a high-resolution photo of your room or property.</p>
            </motion.div>
            <UploadZone onFileSelect={handleFileSelect} />
            <button
              onClick={() => setStage("landing")}
              className="mt-12 text-sm font-semibold tracking-wide transition-colors"
              style={{ color: 'var(--text-muted)' }}
              onMouseOver={(e) => e.target.style.color = 'var(--text-heading)'}
              onMouseOut={(e) => e.target.style.color = 'var(--text-muted)'}
            >
              &larr; Cancel
            </button>
          </motion.div>
        )}

        {/* CONFIGURE */}
        {stage === "configure" && (
          <motion.div
            key="configure"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col min-h-screen"
          >
            {/* Header */}
            <header className="px-8 py-4 border-b glass flex justify-between items-center z-20"
              style={{ borderColor: 'var(--border-subtle)' }}
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center font-bold text-white"
                  style={{ background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))', boxShadow: '0 4px 12px var(--glow-primary)' }}
                >V</div>
                <h1 className="text-lg font-bold tracking-tight" style={{ color: 'var(--text-heading)' }}>
                  Visionary <span className="font-light" style={{ color: 'var(--text-muted)' }}>Suite</span>
                </h1>
              </div>
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-success)' }} />
                  <span className="text-[11px] font-medium" style={{ color: 'var(--text-muted)' }}>Design Engine Online</span>
                </div>
                <button onClick={() => setStage("upload")} className="text-xs font-semibold transition-colors"
                  style={{ color: 'var(--text-muted)' }}
                  onMouseOver={(e) => e.target.style.color = 'var(--text-heading)'}
                  onMouseOut={(e) => e.target.style.color = 'var(--text-muted)'}
                >Discard</button>
              </div>
            </header>

            <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-0 overflow-hidden">
              {/* Left: Preview Panel */}
              <div className="lg:col-span-7 p-8 flex items-center justify-center relative overflow-hidden"
                style={{ background: 'var(--bg-surface)' }}
              >
                <motion.div 
                  initial={{ scale: 0.95, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="relative max-w-full max-h-full rounded-2xl overflow-hidden shadow-xl"
                  style={{ border: '1px solid var(--border-subtle)' }}
                >
                  <img src={previewUrl} alt="Preview" className="w-auto h-auto max-h-[70vh] object-contain" />
                  
                  {/* Detections Layer */}
                  <AnimatePresence>
                    {detections.map((d, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="absolute rounded-lg pointer-events-none"
                        style={{
                           left: `${(d.bbox[0] / 1000) * 100}%`,
                           top: `${(d.bbox[1] / 1000) * 100}%`,
                           width: `${((d.bbox[2] - d.bbox[0]) / 1000) * 100}%`,
                           height: `${((d.bbox[3] - d.bbox[1]) / 1000) * 100}%`,
                           border: '1px solid rgba(167,139,250,0.4)',
                           background: 'rgba(167,139,250,0.08)',
                        }}
                      >
                        <span className="absolute -top-5 left-0 px-1.5 py-0.5 text-[var(--text-heading)] text-[8px] font-bold uppercase tracking-tighter rounded-sm"
                          style={{ background: 'var(--accent-primary)' }}
                        >
                          {d.label}
                        </span>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </motion.div>

                {/* Status Float */}
                <div className="absolute bottom-8 left-8 flex gap-3">
                  <div className="px-4 py-2 glass rounded-2xl flex items-center gap-3"
                    style={{ border: '1px solid var(--border-subtle)' }}
                  >
                    <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs"
                      style={{ background: 'rgba(167,139,250,0.1)', border: '1px solid rgba(167,139,250,0.2)' }}
                    >🏠</div>
                    <div>
                      <p className="text-[10px] font-semibold uppercase" style={{ color: 'var(--text-muted)' }}>Room Analysis</p>
                      <p className="text-xs font-bold capitalize" style={{ color: 'var(--text-heading)' }}>{vlmAnalysis?.current_style || 'Detecting...'}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right: Controls Panel */}
              <div className="lg:col-span-5 border-l p-10 overflow-y-auto"
                style={{ background: 'var(--bg-card)', borderColor: 'var(--border-subtle)' }}
              >
                <div className="max-w-md mx-auto space-y-10">
                  <div>
                    <span className="badge badge-active mb-4">Configuration</span>
                    <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--text-heading)' }}>Customize Your Design</h2>
                    <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>Choose a style, set your budget, and describe your vision.</p>
                  </div>

                  {/* Input Groups */}
                  <section className="space-y-8">
                    <div className="space-y-3">
                      <label className="text-[11px] font-semibold uppercase tracking-wide flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                        <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-primary)' }} />
                        Describe Your Vision
                      </label>
                      <textarea
                        className="w-full h-24 rounded-xl p-4 text-sm transition-all outline-none resize-none"
                        style={{ 
                          background: 'var(--bg-surface)', 
                          border: '1px solid var(--border-subtle)',
                          color: 'var(--text-heading)',
                        }}
                        placeholder="e.g. Minimalist japandi style with walnut textures and warm ambient lighting..."
                        value={localPrompt}
                        onChange={(e) => handlePromptUpdate(e.target.value)}
                        onFocus={(e) => { e.target.style.borderColor = 'var(--accent-primary)'; e.target.style.boxShadow = '0 0 0 3px var(--glow-primary)'; }}
                        onBlur={(e) => { e.target.style.borderColor = 'var(--border-subtle)'; e.target.style.boxShadow = 'none'; }}
                      />
                    </div>

                    <BudgetInput value={localBudget} onChange={handleBudgetUpdate} />
                    
                    {budgetAllocation && (
                      <div className="pt-2">
                        <label className="text-[11px] font-semibold uppercase tracking-wide mb-4 block" style={{ color: 'var(--text-muted)' }}>Budget Breakdown</label>
                        <BudgetAllocationChart data={budgetAllocation} />
                      </div>
                    )}

                    <div className="space-y-3">
                      <label className="text-[11px] font-semibold uppercase tracking-wide flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                        <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-primary)' }} />
                        Rendering Quality
                      </label>
                      <div className="grid grid-cols-2 gap-3">
                        <button
                          onClick={() => setLocalUseSD(false)}
                          className="py-3 px-4 rounded-xl text-[11px] font-semibold tracking-wide transition-all cursor-pointer"
                          style={{
                            background: !localUseSD ? 'rgba(167,139,250,0.12)' : 'var(--bg-surface)',
                            border: `1px solid ${!localUseSD ? 'rgba(167,139,250,0.4)' : 'var(--border-subtle)'}`,
                            color: !localUseSD ? 'var(--accent-primary)' : 'var(--text-muted)',
                            boxShadow: !localUseSD ? '0 4px 12px var(--glow-primary)' : 'none',
                          }}
                        >
                          ⚡ Fast Mode
                        </button>
                        <button
                          onClick={() => setLocalUseSD(true)}
                          className="py-3 px-4 rounded-xl text-[11px] font-semibold tracking-wide transition-all cursor-pointer"
                          style={{
                            background: localUseSD ? 'rgba(249,168,212,0.12)' : 'var(--bg-surface)',
                            border: `1px solid ${localUseSD ? 'rgba(249,168,212,0.4)' : 'var(--border-subtle)'}`,
                            color: localUseSD ? '#db2777' : 'var(--text-muted)',
                            boxShadow: localUseSD ? '0 4px 12px var(--glow-secondary)' : 'none',
                          }}
                        >
                          💎 HD Quality
                        </button>
                      </div>
                    </div>
                  </section>

                  {/* Footer Action */}
                  <div className="pt-6" style={{ borderTop: '1px solid var(--border-dim)' }}>
                    <button
                      onClick={() => submitRedesign(parseFloat(localBudget), localStyle, localPrompt, localUseSD)}
                      disabled={!localBudget || !vlmAnalysis}
                      className="btn btn-primary w-full h-[56px] text-[15px] rounded-2xl disabled:opacity-30 disabled:grayscale transition-all flex items-center justify-center gap-3"
                    >
                      {!vlmAnalysis ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          Analyzing your property...
                        </>
                      ) : (
                        "Generate Redesign ✨"
                      )}
                    </button>
                    <p className="mt-4 text-center text-[11px]" style={{ color: 'var(--text-muted)' }}>Typically takes 8–40 seconds depending on quality setting.</p>
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

        {/* RESULT */}
        {stage === "result" && result && (
          <motion.div
            key="result"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="relative flex flex-col items-center min-h-screen px-6 py-20 overflow-x-hidden"
          >
            {/* Header */}
            <div className="mb-20 text-center max-w-2xl">
              <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
                <span className="badge badge-active mb-4">Project Complete</span>
                <h1 className="mb-4" style={{ color: 'var(--text-heading)' }}>Your Redesigned Space</h1>
                <p className="text-lg font-light leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                  Redesigned in <span className="font-medium capitalize" style={{ color: 'var(--text-heading)' }}>{result.target_style}</span> style. 
                  Architectural integrity preserved with AI-guided synthesis.
                </p>
                
                {result.pcd_url && (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setShowPcd(true)}
                    className="mt-8 btn btn-secondary glass mx-auto"
                  >
                    <span style={{ color: 'var(--accent-primary)' }}>🥽</span> Enter 3D Room Walkthrough
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

            {/* Spatial Insights */}
            <SpatialInsights 
              depthUrl={result.depth_url}
              vlmAnalysis={result.vlm_analysis}
              reasoning={result.reasoning}
            />

            {/* Discovery Grid */}
            <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-12 items-start mb-32">
              
              {/* Info Column */}
              <div className="lg:col-span-4 space-y-12">
                <div className="glass-card p-8">
                  <h3 className="text-sm font-bold uppercase tracking-wide mb-6 flex items-center gap-2" style={{ color: 'var(--text-heading)' }}>
                    <span className="w-2 h-2 rounded-full" style={{ background: 'var(--accent-primary)' }} />
                    Room Analysis
                  </h3>
                  <div className="space-y-5">
                    {[
                      { label: 'Room Type', value: result.vlm_analysis?.room_type },
                      { label: 'Current Style', value: result.vlm_analysis?.current_style },
                      { label: 'Room Size', value: result.vlm_analysis?.room_size_estimate },
                      { label: 'Lighting', value: result.vlm_analysis?.natural_light }
                    ].map(item => (
                      <div key={item.label} className="flex justify-between items-end pb-2" style={{ borderBottom: '1px solid var(--border-dim)' }}>
                        <span className="text-[11px] font-medium uppercase" style={{ color: 'var(--text-muted)' }}>{item.label}</span>
                        <span className="text-xs font-bold capitalize" style={{ color: 'var(--text-heading)' }}>{item.value || 'N/A'}</span>
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
                &larr; Start New Project
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
