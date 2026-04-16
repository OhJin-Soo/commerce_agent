# Commerce Agent — 설계 문서

## 개요

외부 웹 데이터를 내부 커머스 DB로 축적하는 에이전트.
사용자가 질문하면 로컬 DB를 먼저 조회하고, 데이터가 없거나 낡았으면 웹검색 → 파싱 → DB 저장 후 응답한다.
시간이 지날수록 DB에 데이터가 쌓여 웹검색 의존도가 낮아지는 구조.

에이전트 워크플로우는 **LangGraph** `StateGraph`로 구현한다.

---

## 시스템 구조

```
User Query
    ↓
FastAPI (POST /query)
    ↓
LangGraph CommerceGraph
    ↓
[classify_intent]
    ├─ structured  →  [check_freshness]  ──fresh──→  [run_sql]  ──→  [generate_response]
    │                       └──stale──→  [web_search]  ──→  (아래 공통 경로)
    ├─ freshness   →  [web_search]  →  [crawl]  →  [normalize]  →  [upsert_db]  →  [run_sql]  →  [generate_response]
    └─ interpret   →  [check_freshness]  ──fresh──→  [generate_response]
                            └──stale──→  [web_search]  →  [crawl]  →  [normalize]  →  [upsert_db]  →  [generate_response]
```

---

## LangGraph 워크플로우

### AgentState

그래프 전체를 흐르는 공유 상태. 모든 노드는 이 state를 읽고 업데이트한다.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    query:           str
    intent:          str | None          # "structured" | "freshness" | "interpret"
    is_fresh:        bool | None         # DB 데이터 신선도
    search_results:  list[dict]          # 웹검색 원문
    crawled_pages:   list[dict]          # 상세 페이지 크롤링 결과
    products:        list[dict]          # normalize된 상품 목록
    sql_query:       str | None          # 생성된 SQL
    sql_results:     list[dict]          # SQL 실행 결과
    response:        str | None          # 최종 응답
    messages:        Annotated[list, add_messages]  # 대화 히스토리
```

### 노드 목록

| 노드 | 파일 | 역할 |
|---|---|---|
| `classify_intent` | `agent/nodes/intent.py` | LLM으로 질의 유형 분류 |
| `check_freshness` | `agent/nodes/freshness.py` | DB TTL 기반 신선도 확인 |
| `run_sql` | `agent/nodes/sql.py` | NL→SQL 변환 + 실행 |
| `web_search` | `agent/nodes/search.py` | Tavily API 검색 |
| `crawl` | `agent/nodes/crawler.py` | 상품 상세 페이지 크롤링 |
| `normalize` | `agent/nodes/normalizer.py` | LLM으로 상품 구조화 |
| `upsert_db` | `agent/nodes/db_writer.py` | DB upsert |
| `generate_response` | `agent/nodes/responder.py` | 최종 응답 생성 |

### 그래프 정의 (`agent/graph.py`)

```python
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    classify_intent, check_freshness, run_sql,
    web_search, crawl, normalize, upsert_db, generate_response,
)

def route_by_intent(state: AgentState) -> str:
    return state["intent"]  # "structured" | "freshness" | "interpret"

def route_by_freshness(state: AgentState) -> str:
    return "fresh" if state["is_fresh"] else "stale"

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("classify_intent",    classify_intent)
    g.add_node("check_freshness",    check_freshness)
    g.add_node("run_sql",            run_sql)
    g.add_node("web_search",         web_search)
    g.add_node("crawl",              crawl)
    g.add_node("normalize",          normalize)
    g.add_node("upsert_db",          upsert_db)
    g.add_node("generate_response",  generate_response)

    g.set_entry_point("classify_intent")

    g.add_conditional_edges("classify_intent", route_by_intent, {
        "structured":  "check_freshness",
        "freshness":   "web_search",
        "interpret":   "check_freshness",
    })

    g.add_conditional_edges("check_freshness", route_by_freshness, {
        "fresh": "run_sql",
        "stale": "web_search",
    })

    g.add_edge("web_search",   "crawl")
    g.add_edge("crawl",        "normalize")
    g.add_edge("normalize",    "upsert_db")
    g.add_edge("upsert_db",    "run_sql")
    g.add_edge("run_sql",      "generate_response")
    g.add_edge("generate_response", END)

    return g.compile()

commerce_graph = build_graph()
```

---

## 디렉터리 구조

```
commerce_agent/
├── main.py                      # FastAPI 앱 진입점
├── agent/
│   ├── graph.py                 # StateGraph 조립 및 compile
│   ├── state.py                 # AgentState TypedDict
│   └── nodes/
│       ├── intent.py            # classify_intent 노드
│       ├── freshness.py         # check_freshness 노드
│       ├── sql.py               # run_sql 노드 (NL→SQL + 실행)
│       ├── search.py            # web_search 노드
│       ├── crawler.py           # crawl 노드
│       ├── normalizer.py        # normalize 노드
│       ├── db_writer.py         # upsert_db 노드
│       └── responder.py         # generate_response 노드
├── db/
│   ├── models.py                # SQLAlchemy ORM 모델
│   ├── session.py               # DB 세션 관리
│   ├── crud.py                  # upsert / 조회 함수
│   └── freshness.py             # TTL 체크 로직
└── schemas/
    └── product.py               # Pydantic 스키마 (NormalizedProduct)
```

---

## DB 스키마

### raw_search_results

웹검색 원문 보관. 파싱 재시도나 감사 추적용.

```sql
CREATE TABLE raw_search_results (
    id          BIGSERIAL PRIMARY KEY,
    query       TEXT NOT NULL,
    title       TEXT,
    snippet     TEXT,
    url         TEXT NOT NULL,
    source      TEXT,               -- 'tavily', 'naver' 등
    raw_text    TEXT,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ON raw_search_results (query, fetched_at DESC);
```

### normalized_products

실제 서비스 응답에 쓰이는 정제된 상품 정보.

```sql
CREATE TABLE normalized_products (
    id                BIGSERIAL PRIMARY KEY,
    source_site       TEXT NOT NULL,
    source_product_id TEXT,
    source_url        TEXT NOT NULL,
    name              TEXT NOT NULL,
    brand             TEXT,
    category          TEXT,
    price             NUMERIC(12, 0),
    currency          TEXT DEFAULT 'KRW',
    rating            NUMERIC(3, 2),
    review_count      INT,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (source_site, source_url)
);

CREATE INDEX ON normalized_products (category, price);
CREATE INDEX ON normalized_products (brand, rating DESC);
```

### product_facts

EAV 테이블. 색상·용량·배송방식 등 카테고리마다 다른 속성을 유연하게 저장.

```sql
CREATE TABLE product_facts (
    id              BIGSERIAL PRIMARY KEY,
    product_id      BIGINT NOT NULL REFERENCES normalized_products(id) ON DELETE CASCADE,
    attribute_name  TEXT NOT NULL,
    attribute_value TEXT NOT NULL,

    UNIQUE (product_id, attribute_name)
);

CREATE INDEX ON product_facts (attribute_name, attribute_value);
```

### source_provenance

특정 필드값의 출처와 신뢰도 추적.

```sql
CREATE TABLE source_provenance (
    id               BIGSERIAL PRIMARY KEY,
    entity_type      TEXT NOT NULL,
    entity_id        BIGINT NOT NULL,
    source_url       TEXT NOT NULL,
    extracted_field  TEXT NOT NULL,
    extracted_value  TEXT NOT NULL,
    confidence       NUMERIC(3, 2),
    extracted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## TTL 정책

```python
TTL = {
    "price":        timedelta(days=1),
    "stock":        timedelta(days=1),
    "rating":       timedelta(days=3),
    "review_count": timedelta(days=3),
    "product_info": timedelta(days=30),
}
```

`check_freshness` 노드에서 `updated_at`과 비교해 `is_fresh`를 결정한다.

---

## 핵심 노드 설계

### `classify_intent` (`nodes/intent.py`)

```python
async def classify_intent(state: AgentState) -> AgentState:
    # LLM structured output으로 QueryIntent 반환
    # state["intent"] 업데이트
```

| intent | 예시 |
|---|---|
| `structured` | "5만원 이하 이어폰", "나이키 평점 높은 순" |
| `freshness` | "지금 할인 중인 제품", "오늘 최저가" |
| `interpret` | "가성비 좋은 러닝화 추천", "이 둘 차이가 뭐야?" |

### `run_sql` (`nodes/sql.py`)

```python
async def run_sql(state: AgentState) -> AgentState:
    # 시스템 프롬프트에 DB 스키마 포함
    # LLM이 SELECT 문 생성
    # SELECT-only 검증 후 실행
    # state["sql_query"], state["sql_results"] 업데이트
```

### `normalize` (`nodes/normalizer.py`)

```python
class NormalizedProduct(BaseModel):
    source_site:   str
    source_url:    str
    name:          str
    brand:         str | None
    category:      str | None
    price:         int | None
    rating:        float | None
    review_count:  int | None
    facts:         dict[str, str] = {}

async def normalize(state: AgentState) -> AgentState:
    # crawled_pages를 LLM structured output으로 NormalizedProduct 리스트로 변환
    # state["products"] 업데이트
```

---

## API 엔드포인트

```
POST /query
    body: { "query": "5만원 이하 무선 이어폰 추천해줘" }
    → commerce_graph.ainvoke({"query": ..., "messages": [...]})

GET /products
    query params: category, brand, max_price, min_rating, sort_by

GET /products/{id}
    → 상품 상세 + product_facts

POST /refresh/{id}
    → is_fresh=False로 강제 설정 후 그래프 재실행
```

---

## 구현 순서 (Phase)

### Phase 1 — DB + 기본 파이프라인
- [ ] SQLAlchemy 모델 및 마이그레이션 (Alembic)
- [ ] 웹검색 연동 (Tavily API)
- [ ] `normalize` 노드 (LLM structured output)
- [ ] `upsert_db` 노드
- [ ] 단일 카테고리(이어폰)로 `web_search → crawl → normalize → upsert_db` end-to-end 검증

### Phase 2 — LangGraph 워크플로우
- [ ] `AgentState` 및 `StateGraph` 기본 골격
- [ ] `classify_intent` 노드 + 조건부 엣지
- [ ] `check_freshness` 노드 + TTL 로직
- [ ] `run_sql` 노드
- [ ] `generate_response` 노드
- [ ] FastAPI `/query` 엔드포인트 연결

### Phase 3 — 품질 개선
- [ ] LangGraph `MemorySaver`로 멀티턴 대화 상태 유지
- [ ] Redis 캐시 (자주 조회되는 SQL 결과)
- [ ] pgvector 연동 (`interpret` 경로 RAG)
- [ ] 커버리지 확대 (무신사, 올리브영)

---

## 기술 스택

| 역할 | 라이브러리 |
|---|---|
| API | FastAPI + uvicorn |
| 에이전트 워크플로우 | LangGraph |
| DB | PostgreSQL + SQLAlchemy (async) + Alembic |
| LLM | Anthropic SDK (`claude-sonnet-4-6`) |
| 웹검색 | Tavily API |
| 크롤링 | httpx + BeautifulSoup4 |
| 캐시 (Phase 3) | Redis (aioredis) |
| 벡터 (Phase 3) | pgvector |
| 테스트 | pytest + pytest-asyncio |
