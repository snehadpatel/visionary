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
      className="w-full max-w-3xl glass-card"
    >
      <div className="flex items-center gap-3 mb-5">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-sm">
          ✦
        </div>
        <div>
          <h3 className="text-lg font-bold">Refine your design</h3>
          <p className="text-xs text-neutral-500">Chat with Visionary's VLM to adjust your room</p>
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
              Your room is ready! Want to refine anything? Try: 'remove the plants', 'make it darker', 'add a bookshelf'
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
            <div className="chat-bubble-assistant border-indigo-500/30 bg-indigo-500/5">
              <p className="text-sm leading-relaxed">
                {chatTokens}
                <span className="inline-block w-1 h-4 bg-indigo-400 ml-1 animate-pulse" />
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
          placeholder='e.g. "add more plants" or "darker walls"'
          className="flex-1 bg-neutral-900 border border-neutral-700 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 transition-all placeholder-neutral-600"
          id="chat-input"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim()}
          className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl text-sm font-semibold transition-colors cursor-pointer shadow-lg shadow-indigo-500/20"
        >
          Send
        </button>
      </div>
    </motion.div>
  )
}
