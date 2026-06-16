import { useState, useCallback } from "react"
import { sendChatMessage } from "../api/client"

export default function useChat(jobId) {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  const send = useCallback(async (text) => {
    if (!text.trim() || loading || !jobId) return

    setMessages((prev) => [...prev, { role: "user", content: text }])
    setLoading(true)

    try {
      const { response } = await sendChatMessage(jobId, text)
      setMessages((prev) => [...prev, { role: "assistant", content: response }])
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, something went wrong. Please try again." },
      ])
    }

    setLoading(false)
  }, [jobId, loading])

  const clearChat = useCallback(() => {
    setMessages([])
  }, [])

  return { messages, loading, send, clearChat }
}
