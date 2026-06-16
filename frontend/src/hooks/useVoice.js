import { useState, useRef, useCallback, useEffect } from "react"

/**
 * Voice interaction hook for Visionary Live mode.
 * Uses Web Speech API for STT and TTS (English only).
 */
export default function useVoice({ onTranscript = null, lang = "en-US" } = {}) {
  const [isListening, setIsListening] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [interimTranscript, setInterimTranscript] = useState("")
  const [supported, setSupported] = useState(true)

  const recognitionRef = useRef(null)
  const utteranceQueueRef = useRef([])
  const speakingRef = useRef(false)

  // Initialize speech recognition
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      setSupported(false)
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = lang

    recognition.onresult = (event) => {
      let interim = ""
      let final = ""

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          final += result[0].transcript
        } else {
          interim += result[0].transcript
        }
      }

      setInterimTranscript(interim)

      if (final) {
        setTranscript(final.trim())
        if (onTranscript) {
          onTranscript(final.trim())
        }
      }
    }

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error)
      if (event.error === "not-allowed") {
        setSupported(false)
      }
    }

    recognition.onend = () => {
      // Auto-restart if still in listening mode
      if (recognitionRef.current?._shouldListen) {
        try {
          recognition.start()
        } catch (e) {
          // Already started
        }
      } else {
        setIsListening(false)
      }
    }

    recognitionRef.current = recognition

    return () => {
      recognition.abort()
    }
  }, [lang, onTranscript])

  // Start listening
  const startListening = useCallback(() => {
    if (!recognitionRef.current || !supported) return
    try {
      recognitionRef.current._shouldListen = true
      recognitionRef.current.start()
      setIsListening(true)
      setInterimTranscript("")
    } catch (e) {
      // Already started
    }
  }, [supported])

  // Stop listening
  const stopListening = useCallback(() => {
    if (!recognitionRef.current) return
    recognitionRef.current._shouldListen = false
    recognitionRef.current.stop()
    setIsListening(false)
    setInterimTranscript("")
  }, [])

  // Toggle listening
  const toggleListening = useCallback(() => {
    if (isListening) {
      stopListening()
    } else {
      startListening()
    }
  }, [isListening, startListening, stopListening])

  // Speak text using TTS
  const speak = useCallback(
    (text) => {
      if (!window.speechSynthesis || !text) return

      // Cancel any ongoing speech
      window.speechSynthesis.cancel()

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = lang
      utterance.rate = 1.05
      utterance.pitch = 1.0

      // Try to use a good English voice
      const voices = window.speechSynthesis.getVoices()
      const preferred = voices.find(
        (v) =>
          v.lang.startsWith("en") &&
          (v.name.includes("Samantha") ||
            v.name.includes("Google") ||
            v.name.includes("Microsoft") ||
            v.name.includes("Premium"))
      )
      if (preferred) {
        utterance.voice = preferred
      }

      utterance.onstart = () => {
        setIsSpeaking(true)
        speakingRef.current = true
      }
      utterance.onend = () => {
        setIsSpeaking(false)
        speakingRef.current = false
      }
      utterance.onerror = () => {
        setIsSpeaking(false)
        speakingRef.current = false
      }

      window.speechSynthesis.speak(utterance)
    },
    [lang]
  )

  // Stop speaking
  const stopSpeaking = useCallback(() => {
    window.speechSynthesis?.cancel()
    setIsSpeaking(false)
    speakingRef.current = false
  }, [])

  // Cleanup
  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel()
      recognitionRef.current?.abort()
    }
  }, [])

  return {
    isListening,
    isSpeaking,
    transcript,
    interimTranscript,
    supported,
    startListening,
    stopListening,
    toggleListening,
    speak,
    stopSpeaking,
  }
}
