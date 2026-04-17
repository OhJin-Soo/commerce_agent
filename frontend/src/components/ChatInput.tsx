import { useState, type KeyboardEvent } from "react"

interface Props {
  onSend: (text: string) => void
  disabled: boolean
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState("")

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue("")
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="flex items-end gap-2 bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm">
      <textarea
        rows={1}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="이어폰 5만원 이하, 이어폰 추천해줘 ..."
        disabled={disabled}
        className="flex-1 resize-none bg-transparent text-sm text-gray-800 placeholder-gray-400 outline-none leading-relaxed max-h-32 disabled:opacity-50"
      />
      <button
        onClick={submit}
        disabled={disabled || !value.trim()}
        className="shrink-0 w-8 h-8 flex items-center justify-center rounded-xl bg-indigo-500 text-white disabled:opacity-40 hover:bg-indigo-600 transition-colors"
      >
        <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
          <path d="M3.105 2.288a.75.75 0 0 0-.826.95l1.953 6.543H13.5a.75.75 0 0 1 0 1.5H4.232l-1.953 6.543a.75.75 0 0 0 .826.95 28.9 28.9 0 0 0 15.293-7.154.75.75 0 0 0 0-1.115A28.9 28.9 0 0 0 3.105 2.288Z" />
        </svg>
      </button>
    </div>
  )
}
