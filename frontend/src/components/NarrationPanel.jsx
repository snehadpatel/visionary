import { useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"

/**
 * AI narration and conversation panel for live mode.
 * Token-by-token text with typewriter effect, voice waveform, and action chips.
 */
export default function NarrationPanel({
  messages = [],
  narration = "",
  narrationDone = true,
  isListening = false,
  isSpeaking = false,
  interimTranscript = "",
  onChipClick,
}) {
  const scrollRef = useRef(null)

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, narration])

  const quickChips = [
    "What style is this room?",
    "Suggest a redesign",
    "What furniture should I change?",
    "Estimate the budget",
  ]

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="live-narration-panel flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
          <span className="text-xs font-bold uppercase tracking-wider text-neutral-400">
            AI Assistant
          </span>
        </div>
        {/* Voice status */}
        <div className="flex items-center gap-2">
          {isListening && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="flex items-center gap-1.5"
            >
              <div className="voice-wave">
                {[...Array(4)].map((_, i) => (
                  <div
                    key={i}
                    className="voice-bar"
                    style={{ animationDelay: `${i * 0.1}s` }}
                  />
                ))}
              </div>
              <span className="text-[10px] text-red-400 font-medium">
                Listening
              </span>
            </motion.div>
          )}
          {isSpeaking && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="flex items-center gap-1.5"
            >
              <div className="voice-wave speaking">
                {[...Array(4)].map((_, i) => (
                  <div
                    key={i}
                    className="voice-bar"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
              <span className="text-[10px] text-indigo-400 font-medium">
                Speaking
              </span>
            </motion.div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-[120px] max-h-[300px]"
      >
        {messages.length === 0 && !narration && (
          <div className="text-center py-6">
            <p className="text-neutral-600 text-sm">
              Point your camera at a room and ask me anything...
            </p>
          </div>
        )}

        <AnimatePresence>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ y: 10, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className={
                msg.role === "user"
                  ? "chat-bubble-user ml-auto"
                  : "chat-bubble-assistant"
              }
            >
              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Live narration (typing effect) */}
        {narration && !narrationDone && (
          <motion.div
            initial={{ y: 10, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="chat-bubble-assistant"
          >
            <p className="text-sm whitespace-pre-wrap">
              {narration}
              <span className="narration-cursor">▊</span>
            </p>
          </motion.div>
        )}

        {/* Interim speech transcript */}
        {interimTranscript && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            className="chat-bubble-user ml-auto opacity-50"
          >
            <p className="text-sm italic">{interimTranscript}...</p>
          </motion.div>
        )}
      </div>

      {/* Quick Action Chips */}
      <div className="px-4 py-3 border-t border-white/5">
        <div className="flex flex-wrap gap-1.5">
          {quickChips.map((chip) => (
            <button
              key={chip}
              onClick={() => onChipClick?.(chip)}
              className="px-3 py-1.5 rounded-full text-[11px] font-medium bg-white/[0.04] border border-white/8 text-neutral-400 hover:text-white hover:border-indigo-500/40 hover:bg-indigo-500/10 transition-all cursor-pointer"
            >
              {chip}
            </button>
          ))}
        </div>
      </div>
    </motion.div>
  )
}
