import type { Message } from "../types"
import ProductGrid from "./ProductGrid"

interface Props {
  message: Message
}

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === "user"

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        {/* 말풍선 */}
        <div
          className={`px-4 py-2.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? "bg-indigo-500 text-white rounded-br-sm"
              : "bg-gray-100 text-gray-800 rounded-bl-sm"
          }`}
        >
          {message.text}
        </div>

        {/* 에러 */}
        {message.error && (
          <p className="mt-1 text-xs text-red-500 px-1">{message.error}</p>
        )}

        {/* 상품 카드 */}
        {message.products && message.products.length > 0 && (
          <div className="w-full mt-1">
            <ProductGrid products={message.products} />
          </div>
        )}
      </div>
    </div>
  )
}
