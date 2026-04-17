export interface ProductRow {
  id: number
  name: string
  brand: string | null
  category: string | null
  price: string | null
  rating: string | null
  review_count: number | null
  source_url: string | null
}

export interface QueryResponse {
  response: string
  intent: string
  category: string | null
  sql_rows: ProductRow[]
  error: string | null
}

export interface Message {
  id: string
  role: "user" | "assistant"
  text: string
  products?: ProductRow[]
  error?: string | null
  intent?: string
}
