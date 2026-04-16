# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the project
uv run python main.py

# Add a dependency
uv add <package>

# Run tests (once added)
uv run pytest

# Run a single test
uv run pytest tests/path/to/test.py::test_name
```

## Project Overview

**commerce-agent** is a search-augmented commerce agent that accumulates external web product data into an internal structured DB. The core loop:

1. User query arrives
2. Check local DB first
3. On miss or stale data → web search
4. Parse and normalize search results
5. Upsert into DB
6. Generate response from structured data (+ LLM for interpretation)
7. Future queries hit DB first

The project is in early/design stage — `main.py` is a stub and no dependencies are wired yet.

## Intended Architecture

| Layer | POC (Phase 1) | Phase 2+ |
|---|---|---|
| API | FastAPI + uvicorn | 동일 |
| Agent workflow | LangGraph | 동일 |
| DB | PostgreSQL + SQLAlchemy (async) + Alembic | 동일 |
| LLM | Ollama (`llama3.1:8b`) | HuggingFace `meta-llama/Llama-3.1-8B-Instruct` (`from_pretrained`) |
| Data source | Kaggle CSV (`kagglehub`) | 동일 |
| Vector (Phase 3) | — | pgvector |
| Cache (Phase 3) | — | Redis |

**Query routing (Intent Router):**
- Structured queries (`"이어폰 5만원 이하"`) → SQL Agent → PostgreSQL
- Category not yet loaded → Kaggle CSV Ingestion → DB Upsert → SQL
- Interpretation/recommendation → LLM response generation

## Database Schema (planned)

- **normalized_products** — structured product rows (product_id, name, brand, category, price, currency, rating, review_count, source_site, source_url, updated_at)
- **product_facts** — EAV table for additional attributes (product_id, attribute_name, attribute_value)
- **source_provenance** — traceability (entity_type, entity_id, source_url, extracted_field, extracted_value, confidence)

## Dependencies

**Phase 1 (POC):**
```bash
uv add fastapi uvicorn
uv add langgraph langchain-core langchain-ollama
uv add kagglehub pandas
uv add sqlalchemy[asyncio] asyncpg alembic
uv add sqlparse
uv add pydantic
uv add --dev pytest pytest-asyncio
```

**Phase 2+ (Ollama 제거, HuggingFace + GPU 서빙으로 교체):**
```bash
uv remove langchain-ollama
uv add langchain-openai   # SGLang / vLLM OpenAI-compatible 엔드포인트용
# SGLang 또는 vLLM은 별도 서버로 실행 (pip install sglang 또는 vllm)
```

## Key Design Decisions

- **Upsert, not insert**: unique key is `(source_site, source_product_id)` or `(normalized_name, brand)`
- **TTL by field type**: price/stock = 1 day, ratings = 3 days, base product info = 30 days
- **LLM is last-mile only**: search → crawl → SQL → LLM for final text generation; avoid routing all logic through LLM
- **2-step product resolution**: search results find candidates; product detail page crawl provides structured attributes
- **SQL agent handles structured queries only** — interpretation/recommendation needs LLM layer on top
