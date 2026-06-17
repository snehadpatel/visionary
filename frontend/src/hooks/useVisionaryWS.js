import { useState, useCallback, useRef, useEffect } from "react"

// Multi-fallback WebSocket strategy for local dev stability
const getWsUrl = () => {
  const envUrl = import.meta.env.VITE_BACKEND_URL;
  if (envUrl) {
    const wsBase = envUrl.replace(/^http/, 'ws');
    return [wsBase.endsWith('/ws') ? wsBase : `${wsBase}/ws`]
  }
  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    // Try primary direct IP, fallback to localhost name
    return ["ws://127.0.0.1:8080/ws", "ws://localhost:8080/ws"]
  }
  return [`${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`]
}

const WS_URLS = getWsUrl()

export default function useVisionaryWS() {
  const [stages, setStages] = useState({ stage: "landing" })
  const [socket, setSocket] = useState(null)
  const [clientId] = useState(() => Math.random().toString(36).substring(7))
  
  // App State
  const [detections, setDetections] = useState([])
  const [vlmAnalysis, setVlmAnalysis] = useState(null)
  const [budgetAllocation, setBudgetAllocation] = useState(null)
  const [pipelineStep, setPipelineStep] = useState(null)
  const [sdPreview, setSdPreview] = useState(null)
  const [matchedProducts, setMatchedProducts] = useState([])
  const [result, setResult] = useState(null)
  const [chatTokens, setChatTokens] = useState("")
  const [messages, setMessages] = useState([])
  const [error, setError] = useState(null)

  // Live Mode State
  const [isLive, setIsLive] = useState(false)
  const [liveScene, setLiveScene] = useState(null)
  const [liveDetections, setLiveDetections] = useState([])
  const [liveNarration, setLiveNarration] = useState("")
  const [liveNarrationDone, setLiveNarrationDone] = useState(true)
  const [quickRedesignPreview, setQuickRedesignPreview] = useState(null)
  const [liveRedesignFrame, setLiveRedesignFrame] = useState(null)

  const socketRef = useRef(null)
  const pingInterval = useRef(null)

  const urlIndexRef = useRef(0)
  const reconnectTimeoutRef = useRef(null)

  const connect = useCallback(() => {
    if (socketRef.current) return
    
    const currentUrl = `${WS_URLS[urlIndexRef.current]}/${clientId}`
    console.log(`📡 Attempting Neural Link: ${currentUrl}`)
    
    const ws = new WebSocket(currentUrl)
    
    ws.onopen = () => {
      console.log("✅ Neural Link Established")
      setSocket(ws)
      socketRef.current = ws
      urlIndexRef.current = 0 // Reset on success
      setError(null)
    }

    ws.onmessage = (event) => {
      try {
        const { type, data } = JSON.parse(event.data)
        
        switch (type) {
          case "connected":
            console.log("🧠 Core Sync Complete:", data.client_id)
            break
          case "detection_result":
            setDetections(data.objects)
            break
          case "vlm_stream":
            setVlmAnalysis(data)
            break
          case "budget_preview":
            setBudgetAllocation(data)
            break
          case "pipeline_step":
            setPipelineStep(data)
            break
          case "sd_preview":
            setSdPreview(data.preview_base64)
            break
          case "products_found":
            setMatchedProducts(data.products)
            break
          case "result_ready":
            setResult(data)
            setStages({ stage: "result" })
            break
          case "chat_token":
            setChatTokens((prev) => prev + data.token)
            break
          case "chat_done":
            setMessages((prev) => [...prev, { role: "assistant", content: data.full_response }])
            setChatTokens("")
            break

          // ─── Live Mode Events ───
          case "live_session_started":
            setIsLive(true)
            break
          case "live_session_stopped":
            setIsLive(false)
            break
          case "live_scene_update":
            setLiveScene(data)
            break
          case "live_detections":
            setLiveDetections(data.objects || [])
            break
          case "live_narration":
            if (data.done) {
              setLiveNarration(data.full_text)
              setLiveNarrationDone(true)
              setMessages((prev) => [...prev, { role: "assistant", content: data.full_text }])
            } else {
              setLiveNarration((prev) => prev + data.token)
              setLiveNarrationDone(false)
            }
            break
          case "quick_redesign_preview":
            setQuickRedesignPreview(data)
            break
          case "live_redesign_frame":
            setLiveRedesignFrame(data.image_b64)
            break

          case "error":
            setError(data.message)
            break
          default:
            break
        }
      } catch (err) {
        console.error("WS Parse Error:", err, event.data)
      }
    }

    ws.onclose = () => {
      console.warn("⚠️ Neural Link Interrupted")
      setSocket(null)
      socketRef.current = null
      if (pingInterval.current) clearInterval(pingInterval.current)
      
      // Rotate through URLs on failure
      urlIndexRef.current = (urlIndexRef.current + 1) % WS_URLS.length
      
      // Retry with backoff
      reconnectTimeoutRef.current = setTimeout(() => {
        connect()
      }, 2000)
    }

    ws.onerror = (err) => {
      console.error("❌ Neural Link Error:", err)
      ws.close()
    }

    // Keep-alive heartbeat
    pingInterval.current = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping", data: {} }))
      }
    }, 5000)
  }, [clientId])

  useEffect(() => {
    connect()
    return () => {
      if (socketRef.current) socketRef.current.close()
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current)
    }
  }, [connect])

  // ─── Standard Mode Actions ───

  const uploadImage = useCallback((file) => {
    if (!socketRef.current) return
    const reader = new FileReader()
    reader.onload = () => {
      socketRef.current.send(JSON.stringify({
        type: "upload_image",
        data: { image_b64: reader.result, filename: file.name }
      }))
    }
    reader.readAsDataURL(file)
  }, [])

  const changeBudget = useCallback((budgetInr, style, prompt) => {
    if (!socketRef.current) return
    socketRef.current.send(JSON.stringify({
      type: "budget_changed",
      data: { budget_inr: budgetInr, style, prompt }
    }))
  }, [])

  const submitRedesign = useCallback((budgetInr, style, prompt, useSD = false) => {
    if (!socketRef.current) return
    socketRef.current.send(JSON.stringify({
      type: "submit_redesign",
      data: { budget_inr: budgetInr, style, prompt, use_sd: useSD }
    }))
    setStages({ stage: "processing" })
  }, [])

  const sendChatMessage = useCallback((message) => {
    if (!socketRef.current) return
    setMessages((prev) => [...prev, { role: "user", content: message }])
    socketRef.current.send(JSON.stringify({
      type: "chat_message",
      data: { message }
    }))
  }, [])

  // ─── Live Mode Actions ───

  const startLive = useCallback(() => {
    if (!socketRef.current) return
    socketRef.current.send(JSON.stringify({ type: "start_live_session", data: {} }))
    setStages({ stage: "live" })
  }, [])

  const stopLive = useCallback(() => {
    if (!socketRef.current) return
    socketRef.current.send(JSON.stringify({ type: "stop_live_session", data: {} }))
    setIsLive(false)
    setLiveScene(null)
    setLiveDetections([])
    setLiveNarration("")
  }, [])

  const sendLiveFrame = useCallback((frameDataUrl) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return
    socketRef.current.send(JSON.stringify({
      type: "live_frame",
      data: { frame_b64: frameDataUrl }
    }))
  }, [])

  const sendVoiceInput = useCallback((text) => {
    if (!socketRef.current || !text) return
    setMessages((prev) => [...prev, { role: "user", content: text }])
    setLiveNarration("")
    setLiveNarrationDone(false)
    socketRef.current.send(JSON.stringify({
      type: "voice_input",
      data: { text }
    }))
  }, [])

  const processHttpFrame = useCallback(async (frameDataUrl, includeRedesign = true) => {
    try {
      const formData = new FormData()
      formData.append("image_b64", frameDataUrl)
      formData.append("style", "luxury")
      formData.append("include_redesign", includeRedesign)

      const backendUrl = import.meta.env.VITE_BACKEND_URL || '';
      const response = await fetch(`${backendUrl}/api/live/frame`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("HTTP Live Frame failed")
      
      const data = await response.json()
      
      // Sync results to state just like WebSocket would
      setLiveDetections(data.detections || [])
      if (data.scene_state) {
        setLiveScene(data.scene_state)
      }
      if (data.redesign_frame) {
        setLiveRedesignFrame(data.redesign_frame)
      }
      
      return data
    } catch (err) {
      console.error("HTTP Live Error:", err)
      setError("HTTP Live Mode failed. Check backend connection.")
    }
  }, [])

  const reset = useCallback(() => {
    setStages({ stage: "landing" })
    setDetections([])
    setVlmAnalysis(null)
    setBudgetAllocation(null)
    setPipelineStep(null)
    setSdPreview(null)
    setMatchedProducts([])
    setResult(null)
    setMessages([])
    setChatTokens("")
    setError(null)
    setIsLive(false)
    setLiveScene(null)
    setLiveDetections([])
    setLiveNarration("")
    setQuickRedesignPreview(null)
    setLiveRedesignFrame(null)
  }, [])

  return {
    stage: stages.stage,
    setStage: (s) => setStages({ stage: s }),
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
  }
}

