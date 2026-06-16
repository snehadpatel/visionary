import { useState, useCallback, useRef, useEffect } from "react"
import { startRedesign, pollStatus } from "../api/client"

export default function useRedesign() {
  const [stage, setStage] = useState("landing")
  const [imageFile, setImageFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [budget, setBudget] = useState("")
  const [style, setStyle] = useState("auto")
  const [prompt, setPrompt] = useState("")
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [result, setResult] = useState(null)
  const intervalRef = useRef(null)

  const handleFileSelect = useCallback((file) => {
    setImageFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setStage("configure")
  }, [])

  const handleSubmit = useCallback(async () => {
    if (!imageFile || !budget) return
    setStage("processing")
    
    try {
      const { job_id } = await startRedesign(
        imageFile,
        prompt,
        parseFloat(budget),
        style
      )
      setJobId(job_id)

      intervalRef.current = setInterval(async () => {
        try {
          const status = await pollStatus(job_id)
          setJobStatus(status)
          if (status.status === "done") {
            clearInterval(intervalRef.current)
            setResult(status.result)
            setStage("result")
          } else if (status.status === "error") {
            clearInterval(intervalRef.current)
            setStage("configure")
          }
        } catch (err) {
          console.error("Polling error:", err)
        }
      }, 2500)
    } catch (err) {
      console.error("Submit error:", err)
      setStage("configure")
    }
  }, [imageFile, budget, style, prompt])

  const reset = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    setStage("landing")
    setImageFile(null)
    setPreviewUrl(null)
    setBudget("")
    setStyle("auto")
    setPrompt("")
    setJobId(null)
    setJobStatus(null)
    setResult(null)
  }, [])

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  return {
    stage, setStage,
    imageFile, previewUrl,
    budget, setBudget,
    style, setStyle,
    prompt, setPrompt,
    jobId, jobStatus,
    result,
    handleFileSelect,
    handleSubmit,
    reset,
  }
}
