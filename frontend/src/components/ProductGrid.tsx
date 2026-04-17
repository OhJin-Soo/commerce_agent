import type { ProductRow } from "../types"

interface Props {
  products: ProductRow[]
}

export default function ProductGrid({ products }: Props) {
  if (products.length === 0) return null

  return (
    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
      {products.map((p) => (
        <div
          key={p.id}
          className="bg-white border border-gray-200 rounded-xl p-3 shadow-sm text-left"
        >
          {p.source_url ? (
            <a
              href={p.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-gray-800 hover:text-indigo-600 hover:underline line-clamp-2 leading-snug block"
            >
              {p.name}
            </a>
          ) : (
            <p className="text-sm font-medium text-gray-800 line-clamp-2 leading-snug">
              {p.name}
            </p>
          )}
          <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
            <span className="font-semibold text-indigo-600 text-sm">
              {p.price != null
                ? `₹${Number(p.price).toLocaleString()}`
                : "가격 미상"}
            </span>
            <div className="flex items-center gap-2">
              {p.rating != null && (
                <span className="flex items-center gap-0.5">
                  <span className="text-yellow-400">★</span>
                  {p.rating}
                </span>
              )}
              {p.review_count != null && (
                <span>{p.review_count.toLocaleString()}개 리뷰</span>
              )}
            </div>
          </div>
          {p.category && (
            <span className="mt-1 inline-block text-xs text-gray-400">
              {p.category}
            </span>
          )}
        </div>
      ))}
    </div>
  )
}
