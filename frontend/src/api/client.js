const BASE = "/api"

export async function startRedesign(imageFile, prompt, budgetInr, style) {
  const form = new FormData()
  form.append("image", imageFile)
  form.append("prompt", prompt || "")
  form.append("budget_inr", budgetInr)
  form.append("style", style)
  const res = await fetch(`${BASE}/redesign`, { method: "POST", body: form })
  return res.json()
}

export async function pollStatus(jobId) {
  const res = await fetch(`${BASE}/status/${jobId}`)
  return res.json()
}

export async function sendChatMessage(jobId, message) {
  const form = new FormData()
  form.append("message", message)
  const res = await fetch(`${BASE}/chat/${jobId}`, { method: "POST", body: form })
  return res.json()
}
