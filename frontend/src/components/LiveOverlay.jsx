import { useRef, useEffect, useState } from "react"

/**
 * Canvas overlay for real-time bounding box annotations.
 * Draws animated detection boxes over the camera feed.
 */
export default function LiveOverlay({ detections = [], videoRef, width, height }) {
  const canvasRef = useRef(null)
  const prevDetectionsRef = useRef([])
  const animFrameRef = useRef(null)

  // Color map for different object types
  const colorMap = {
    sofa: "#818cf8",
    chair: "#34d399",
    table: "#fbbf24",
    bed: "#f472b6",
    lamp: "#a78bfa",
    tv: "#60a5fa",
    default: "#6366f1",
  }

  const getColor = (label) => {
    const key = Object.keys(colorMap).find((k) => label.toLowerCase().includes(k))
    return key ? colorMap[key] : colorMap.default
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    canvas.width = width || 640
    canvas.height = height || 480

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      detections.forEach((det, i) => {
        const [x1, y1, x2, y2] = det.bbox
        const color = getColor(det.label)
        const w = ((x2 - x1) / 1000) * canvas.width
        const h = ((y2 - y1) / 1000) * canvas.height
        const x = (x1 / 1000) * canvas.width
        const y = (y1 / 1000) * canvas.height

        const cornerLen = Math.min(w, h) * 0.2
        const lineWidth = 2.5

        // Semi-transparent fill
        ctx.fillStyle = color + "10"
        ctx.fillRect(x, y, w, h)

        // Animated scanner corners
        ctx.strokeStyle = color
        ctx.lineWidth = lineWidth
        ctx.lineCap = "round"

        // Top-left corner
        ctx.beginPath()
        ctx.moveTo(x, y + cornerLen)
        ctx.lineTo(x, y)
        ctx.lineTo(x + cornerLen, y)
        ctx.stroke()

        // Top-right corner
        ctx.beginPath()
        ctx.moveTo(x + w - cornerLen, y)
        ctx.lineTo(x + w, y)
        ctx.lineTo(x + w, y + cornerLen)
        ctx.stroke()

        // Bottom-left corner
        ctx.beginPath()
        ctx.moveTo(x, y + h - cornerLen)
        ctx.lineTo(x, y + h)
        ctx.lineTo(x + cornerLen, y + h)
        ctx.stroke()

        // Bottom-right corner
        ctx.beginPath()
        ctx.moveTo(x + w - cornerLen, y + h)
        ctx.lineTo(x + w, y + h)
        ctx.lineTo(x + w, y + h - cornerLen)
        ctx.stroke()

        // Dashed border
        ctx.setLineDash([4, 4])
        ctx.strokeStyle = color + "40"
        ctx.lineWidth = 1
        ctx.strokeRect(x, y, w, h)
        ctx.setLineDash([])

        // Label badge
        const label = `${det.label} ${Math.round(det.confidence * 100)}%`
        ctx.font = "bold 11px Inter, system-ui, sans-serif"
        const textWidth = ctx.measureText(label).width

        const badgeX = x
        const badgeY = y - 22
        const badgeW = textWidth + 12
        const badgeH = 20
        const badgeR = 6

        // Badge background
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.roundRect(badgeX, badgeY, badgeW, badgeH, badgeR)
        ctx.fill()

        // Badge text
        ctx.fillStyle = "#ffffff"
        ctx.fillText(label, badgeX + 6, badgeY + 14)
      })

      animFrameRef.current = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [detections, width, height])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 10 }}
    />
  )
}
