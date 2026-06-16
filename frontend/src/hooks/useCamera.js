import { useState, useRef, useCallback, useEffect } from "react"

/**
 * Camera management hook for Visionary Live mode.
 * Handles getUserMedia, frame capture, and camera switching.
 */
export default function useCamera({
  width = 640,
  height = 480,
  captureIntervalMs = 2000,
  quality = 0.6,
  onFrame = null,
} = {}) {
  const [isActive, setIsActive] = useState(false)
  const [hasPermission, setHasPermission] = useState(null) // null = unknown
  const [error, setError] = useState(null)
  const [facingMode, setFacingMode] = useState("environment") // back cam

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const intervalRef = useRef(null)

  // Start the camera
  const startCamera = useCallback(async () => {
    try {
      setError(null)
      const constraints = {
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 }, // Let the browser scale down if needed
          height: { ideal: 720 },
        },
        audio: false,
      }

      let stream;
      try {
        console.log("[useCamera] Requesting stream with ideal constraints:", constraints)
        stream = await navigator.mediaDevices.getUserMedia(constraints)
      } catch (err) {
        console.warn("[useCamera] Ideal constraints failed, falling back to basic video: true")
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      }
      
      streamRef.current = stream
      setHasPermission(true)
      setIsActive(true)

      // Attach to video element if ref is set
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }

      // Start frame capture interval
      if (onFrame) {
        intervalRef.current = setInterval(() => {
          captureFrame()
        }, captureIntervalMs)
      }
    } catch (err) {
      console.error("Camera error:", err)
      setHasPermission(false)
      
      let msg = "Camera access denied"
      if (err.name === "NotAllowedError") msg = "Permission denied. Please check your browser settings."
      if (err.name === "NotFoundError") msg = "No camera found on this device."
      if (err.name === "NotReadableError") msg = "Camera is already in use by another app."
      if (!window.isSecureContext) msg = "HTTPS is required for mobile camera access."
      
      setError(msg)
    }
  }, [facingMode, width, height, captureIntervalMs, onFrame])

  // Stop the camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setIsActive(false)
  }, [])

  // Capture a single frame as base64 JPEG
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")

    canvas.width = video.videoWidth || width
    canvas.height = video.videoHeight || height
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    const dataUrl = canvas.toDataURL("image/jpeg", quality)

    if (onFrame) {
      onFrame(dataUrl)
    }

    return dataUrl
  }, [width, height, quality, onFrame])

  // Toggle front/back camera
  const toggleCamera = useCallback(() => {
    const newMode = facingMode === "environment" ? "user" : "environment"
    setFacingMode(newMode)
    if (isActive) {
      stopCamera()
      // Restart will happen due to facingMode change
    }
  }, [facingMode, isActive, stopCamera])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [stopCamera])

  // Restart camera when facing mode changes while active
  useEffect(() => {
    if (isActive) {
      stopCamera()
      startCamera()
    }
  }, [facingMode])

  return {
    videoRef,
    canvasRef,
    isActive,
    hasPermission,
    error,
    facingMode,
    startCamera,
    stopCamera,
    captureFrame,
    toggleCamera,
  }
}
