import { useState, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import useCamera from "../hooks/useCamera"
import useVoice from "../hooks/useVoice"
import LiveOverlay from "./LiveOverlay"
import SceneInfoCards from "./SceneInfoCards"
import NarrationPanel from "./NarrationPanel"

/**
 * LiveMode — Full-screen cinematic live camera experience.
 * 
 * The hero component of Visionary Live. Shows a real-time camera feed
 * with AI-powered bounding boxes, SceneNet info cards, voice interaction,
 * and an AI narration panel.
 */
export default function LiveMode({
  liveScene,
  liveDetections,
  liveNarration,
  liveNarrationDone,
  liveRedesignFrame,
  messages,
  isLive,
  processHttpFrame,
  submitRedesign,
  onExit,
}) {
  const [showPanel, setShowPanel] = useState(true)
  const [showStyles, setShowStyles] = useState(false)
  const [showRedesign, setShowRedesign] = useState(true) // Toggle live redesigned frame overlay
  const [useSD, setUseSD] = useState(false) // Toggle between High-Speed (UNet) and High-Quality (SD)

  const handleRedesign = useCallback((style) => {
    // Trigger the actual generation pipeline from the live frame
    // Pass the useSD flag to determine which neural engine to run
    submitRedesign(30000, style, "", useSD)
    setShowStyles(false)
  }, [submitRedesign, useSD])

  const [clientId] = useState(() => `live_${Math.random().toString(36).substr(2, 9)}`)

  // Voice handler over HTTP
  const sendVoiceHttp = async (text) => {
    try {
      const formData = new FormData()
      formData.append("client_id", clientId)
      formData.append("text", text)
      await fetch("/api/interaction/voice", { method: "POST", body: formData })
    } catch (err) {
      console.error("Voice HTTP error:", err)
    }
  }

  // Camera hook — sends frames via HTTP
  const camera = useCamera({
    width: 640,
    height: 480,
    captureIntervalMs: 2000,
    quality: 0.5,
    onFrame: (frame) => processHttpFrame(frame, showRedesign),
  })

  // Voice hook — handles STT/TTS
  const voice = useVoice({
    onTranscript: sendVoiceHttp,
  })

  // Start camera on mount (No WebSocket start needed!)
  useEffect(() => {
    camera.startCamera()
    return () => {
      camera.stopCamera()
    }
  }, [])

  // Auto-speak narration when complete
  useEffect(() => {
    if (liveNarrationDone && liveNarration && !voice.isListening) {
      voice.speak(liveNarration)
    }
  }, [liveNarrationDone, liveNarration])

  const handleChipClick = useCallback(
    (text) => {
      sendVoiceInput(text)
    },
    [sendVoiceInput]
  )

  const handleExit = useCallback(() => {
    camera.stopCamera()
    voice.stopSpeaking()
    onExit?.()
  }, [camera, voice, onExit])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black z-50 overflow-hidden"
    >
      {/* ─── Camera Feed ─── */}
      <div className="absolute inset-0">
        <video
          ref={camera.videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />

        {/* Camera Error State */}
        {!camera.isActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-40 px-6 text-center">
            <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center text-2xl mb-4">
              ⚠️
            </div>
            <h3 className="text-white font-bold mb-2">Camera Access Failed</h3>
            <p className="text-neutral-400 text-sm mb-8">
              {camera.error || "Please ensure you are using HTTPS and have granted camera permissions."}
            </p>
            <button
              onClick={() => camera.startCamera()}
              className="btn-primary px-8 py-3 rounded-xl font-bold"
            >
              Retry Access
            </button>
          </div>
        )}
        {/* Hidden canvas for frame capture */}
        <canvas ref={camera.canvasRef} className="hidden" />

        {/* Bounding Box Overlay */}
        <LiveOverlay
          detections={liveDetections}
          videoRef={camera.videoRef}
          width={640}
          height={480}
        />

        {/* Live Redesign Overlay (Magic Vision) */}
        <AnimatePresence>
          {showRedesign && liveRedesignFrame && (
            <motion.img
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              src={`data:image/jpeg;base64,${liveRedesignFrame}`}
              className="absolute inset-0 w-full h-full object-cover z-0"
              alt="Live Redesign"
            />
          )}
        </AnimatePresence>
      </div>

      {/* ─── Top Bar ─── */}
      <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-5 py-4">
        {/* LIVE Badge */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-xl border border-white/10"
        >
          <div className="w-2 h-2 rounded-full bg-red-500 live-pulse" />
          <span className="text-[11px] font-bold uppercase tracking-wider text-white">
            Live
          </span>
          {liveScene?.frame_count > 0 && (
            <span className="text-[10px] text-neutral-500">
              #{liveScene.frame_count}
            </span>
          )}
        </motion.div>

        {/* Object Count */}
        {liveDetections.length > 0 && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-xl border border-white/10"
          >
            <span className="text-[11px] text-white font-medium">
              {liveDetections.length} object{liveDetections.length !== 1 && "s"} detected
            </span>
          </motion.div>
        )}

        {/* Exit Button */}
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={handleExit}
          className="w-9 h-9 rounded-full bg-black/60 backdrop-blur-xl border border-white/10 flex items-center justify-center text-white hover:bg-red-500/20 hover:border-red-500/30 transition-colors cursor-pointer"
        >
          ✕
        </motion.button>
      </div>

      {/* ─── Right Panel: SceneNet Info Cards ─── */}
      <div className="absolute top-20 right-4 z-20">
        <SceneInfoCards scene={liveScene} />
      </div>

      {/* ─── Bottom Panel ─── */}
      <div className="absolute bottom-0 left-0 right-0 z-20">
        
        {/* Redesign Styles Popup */}
        <AnimatePresence>
          {showStyles && (
            <motion.div
              initial={{ y: 20, opacity: 0, scale: 0.9 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 20, opacity: 0, scale: 0.9 }}
              className="absolute bottom-24 left-1/2 -translate-x-1/2 flex flex-wrap justify-center gap-2 p-3 bg-black/80 backdrop-blur-2xl rounded-2xl border border-white/10 z-30 w-11/12 max-w-sm"
            >
              <div className="w-full text-center text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-1">
                Select Target Style
              </div>
              
              {/* Quality vs Speed Toggle */}
              <div className="w-full flex items-center justify-between mb-3 px-1 py-2 bg-white/5 rounded-xl border border-white/5">
                <span className="text-[10px] text-neutral-400 font-bold ml-2">Staging Engine</span>
                <div className="flex gap-1 mr-1">
                  <button 
                    onClick={() => setUseSD(false)}
                    className={`px-2 py-1 rounded-md text-[9px] font-bold transition-all ${!useSD ? "bg-[var(--accent-primary)] text-white" : "text-neutral-500"}`}
                  >
                    ⚡ SPEED
                  </button>
                  <button 
                    onClick={() => setUseSD(true)}
                    className={`px-2 py-1 rounded-md text-[9px] font-bold transition-all ${useSD ? "bg-[var(--accent-secondary)] text-white" : "text-neutral-500"}`}
                  >
                    💎 QUALITY
                  </button>
                </div>
              </div>

              {["scandinavian", "industrial", "bohemian", "japandi", "mid-century", "luxury"].map((style) => (
                <button
                  key={style}
                  onClick={() => handleRedesign(style)}
                  className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-medium transition-colors capitalize whitespace-nowrap cursor-pointer"
                >
                  {style.replace("-", " ")}
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Action Buttons */}
        <div className="flex items-center justify-center gap-4 pb-4">
          
          {/* Quick Staging Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowStyles(!showStyles)}
            className={`flex items-center justify-center gap-2 px-5 py-3 rounded-full font-bold text-sm shadow-lg transition-all cursor-pointer ${
              showStyles 
                ? "bg-white text-black shadow-white/20" 
                : "bg-[var(--accent-primary)] text-white shadow-[var(--glow-primary)]"
            }`}
          >
            ✦ Stage Space
          </motion.button>

          {/* Toggle Panel */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setShowPanel(!showPanel)}
            className="live-action-btn"
            title="Toggle chat panel"
          >
            💬
          </motion.button>

          {/* Magic Vision Toggle */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setShowRedesign(!showRedesign)}
            className={`live-action-btn ${showRedesign ? "bg-[var(--accent-primary)] text-white" : ""}`}
            title="Toggle Magic Vision (Live Redesign)"
          >
            ✨
          </motion.button>

          {/* Voice Button — Large, central */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.9 }}
            onClick={voice.toggleListening}
            className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl transition-all cursor-pointer shadow-lg ${
              voice.isListening
                ? "bg-red-500 shadow-red-500/30 live-pulse"
                : "bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-secondary)] shadow-[var(--glow-primary)]"
            }`}
            title={voice.isListening ? "Stop listening" : "Start speaking"}
          >
            {voice.isListening ? "⏹" : "🎤"}
          </motion.button>

          {/* Camera Toggle */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={camera.toggleCamera}
            className="live-action-btn"
            title="Switch camera"
          >
            🔄
          </motion.button>
        </div>

        {/* Narration Panel (collapsible) */}
        <AnimatePresence>
          {showPanel && (
            <motion.div
              initial={{ y: 200, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 200, opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
            >
              <NarrationPanel
                messages={messages}
                narration={liveNarration}
                narrationDone={liveNarrationDone}
                isListening={voice.isListening}
                isSpeaking={voice.isSpeaking}
                interimTranscript={voice.interimTranscript}
                onChipClick={handleChipClick}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ─── Vignette Overlay ─── */}
      <div className="absolute inset-0 pointer-events-none z-10 live-vignette" />
    </motion.div>
  )
}
