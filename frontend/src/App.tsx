import { useEffect, useRef, useState } from "react"
import { fetchRate, postQuery } from "./api"
import ChatInput from "./components/ChatInput"
import ChatMessage from "./components/ChatMessage"
import type { Message } from "./types"

const SUGGESTIONS = [
  "이어폰 5만원 이하",
  "이어폰 추천해줘",
  "스피커 목록",
  "TV 비교해줘",
]

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [rate, setRate] = useState<number>(16.0)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchRate().then(setRate)
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  const send = async (text: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text,
    }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      const data = await postQuery(text)
      const aiMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: data.response || "(응답 없음)",
        products: data.sql_rows ?? [],
        error: data.error ?? null,
        intent: data.intent,
      }
      setMessages((prev) => [...prev, aiMsg])
    } catch (err) {
      const aiMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: "오류가 발생했습니다.",
        error: err instanceof Error ? err.message : String(err),
      }
      setMessages((prev) => [...prev, aiMsg])
    } finally {
      setLoading(false)
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="shrink-0 flex items-center gap-3 px-6 py-4 bg-white border-b border-gray-200">
        <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center">
          <svg viewBox="0 0 20 20" fill="white" className="w-4 h-4">
            <path d="M3 1a1 1 0 0 0 0 2h1.22l.305 1.222a.997.997 0 0 0 .01.042l1.358 5.43-.893.892C3.74 11.846 4.632 14 6.414 14H15a1 1 0 0 0 0-2H6.414l1-1H14a1 1 0 0 0 .894-.553l3-6A1 1 0 0 0 17 3H6.28l-.31-1.243A1 1 0 0 0 5 1H3ZM16 16.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0ZM6.5 18a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z" />
          </svg>
        </div>
        <div>
          <h1 className="text-sm font-semibold text-gray-900 leading-none">Commerce Agent</h1>
          <p className="text-xs text-gray-400 mt-0.5">상품 검색 · 추천</p>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto space-y-4">
          {isEmpty && (
            <div className="text-center pt-16">
              <p className="text-2xl font-semibold text-gray-700 mb-2">무엇을 찾고 계신가요?</p>
              <p className="text-sm text-gray-400 mb-8">상품 검색이나 추천을 물어보세요.</p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="px-4 py-2 text-sm bg-white border border-gray-200 rounded-full text-gray-600 hover:border-indigo-400 hover:text-indigo-600 transition-colors shadow-sm"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} rate={rate} />
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-2xl rounded-bl-sm px-4 py-3 flex gap-1.5 items-center">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="shrink-0 px-4 py-4 bg-gray-50 border-t border-gray-100">
        <div className="max-w-2xl mx-auto">
          <ChatInput onSend={send} disabled={loading} />
          <p className="text-center text-xs text-gray-400 mt-2">
            Enter 전송 · Shift+Enter 줄바꿈
          </p>
        </div>
      </footer>
    </div>
  )
}
