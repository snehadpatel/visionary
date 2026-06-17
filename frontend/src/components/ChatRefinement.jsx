import { useState, useEffect, useRef } from "react"
import { motion } from "framer-motion"

export default function ChatRefinement({ messages, chatTokens, onSendMessage }) {
  const [input, setInput] = useState("")
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, chatTokens])

  const handleSend = () => {
    if (!input.trim()) return
    onSendMessage(input.trim())
    setInput("")
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="w-full max-w-3xl glass-card p-6"
    >
      <div className="flex items-center gap-3 mb-5">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-secondary)] flex items-center justify-center text-sm text-white font-bold">
          ✨
        </div>
        <div>
          <h3 className="text-lg font-bold text-[var(--text-heading)]">Refine Your Redesign</h3>
          <p className="text-xs text-[var(--text-secondary)]">Chat with our AI Interior Designer to adjust your staging</p>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="space-y-4 mb-5 max-h-72 overflow-y-auto pr-2 custom-scrollbar"
      >
        {/* Welcome Message */}
        <div className="flex justify-start">
          <div className="chat-bubble-assistant">
            <p className="text-sm leading-relaxed">
              Your staging render is ready! Want to make any changes? Try: 'add a wooden coffee table', 'change walls to warm beige', 'add some potted plants'.
            </p>
          </div>
        </div>

        {/* Conversation History */}
        {messages.map((m, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div className={m.role === "user" ? "chat-bubble-user" : "chat-bubble-assistant"}>
              <p className="text-sm leading-relaxed">{m.content}</p>
            </div>
          </motion.div>
        ))}

        {/* Live Streaming Tokens */}
        {chatTokens && (
          <div className="flex justify-start">
            <div className="chat-bubble-assistant border-[var(--accent-primary)]/30 bg-[var(--accent-primary)]/5">
              <p className="text-sm leading-relaxed">
                {chatTokens}
                <span className="inline-block w-1 h-4 bg-[var(--accent-primary)] ml-1 animate-pulse" />
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="flex gap-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder='e.g. "add a coffee table" or "beige walls"'
          className="flex-1 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-xl px-4 py-3 text-sm text-[var(--text-heading)] focus:outline-none focus:border-[var(--accent-primary)] focus:ring-1 focus:ring-[var(--accent-primary)]/20 transition-all placeholder-[var(--text-muted)]"
          id="chat-input"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim()}
          className="btn btn-primary disabled:opacity-40 disabled:cursor-not-allowed py-3 px-6 text-sm rounded-xl"
        >
          Send
        </button>
      </div>
    </motion.div>
  )
}
