import type { QueryResponse } from "./types"

export async function postQuery(query: string): Promise<QueryResponse> {
  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail?.detail ?? `HTTP ${res.status}`)
  }
  return res.json()
}

export async function fetchRate(): Promise<number> {
  try {
    const res = await fetch("/rate")
    if (!res.ok) return 16.0
    const data = await res.json()
    return data.inr_to_krw ?? 16.0
  } catch {
    return 16.0
  }
}
