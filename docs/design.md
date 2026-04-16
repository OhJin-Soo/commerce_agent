# Commerce Agent — 설계 문서

## 개요

외부 웹 데이터를 내부 커머스 DB로 축적하는 에이전트.
사용자가 질문하면 로컬 DB를 먼저 조회하고, 데이터가 없거나 낡았으면 웹검색 → 파싱 → DB 저장 후 응답한다.
시간이 지날수록 DB에 데이터가 쌓여 웹검색 의존도가 낮아지는 구조.

에이전트 워크플로우는 **LangGraph** `StateGraph`로 구현한다.

---

## 구현 전략 — 목표 아키텍처 vs Phase 1

### 목표 아키텍처 (Target)

Clean Architecture 4계층 + DDD + 3개 Bounded Context. 설계 문서의 DDD 섹션이 이 구조를 기술한다.

### Phase 1 구현 범위 (KISS)

> **이어폰 단일 카테고리 end-to-end 검증**이 목표.
> 인터페이스, 포트, Domain Event, BC 분리는 도입하지 않는다.

**적용하는 것:**
- FastAPI + LangGraph StateGraph (그래프 구조는 목표와 동일)
- SQLAlchemy 직접 호출 (Repository 인터페이스 없음)
- Pipe-Filter 기반 `IngestionPipeline` (수집 서브파이프라인)
- Pydantic 스키마 (`NormalizedProduct`)

**Phase 2 이후로 미루는 것:**
- `IProductRepository` / `ILLMPort` 인터페이스
- DDD Aggregate / Value Object / Domain Event
- Bounded Context 분리 + ACL
- source_provenance 테이블
- Clean Architecture 레이어 분리

### Phase 1 디렉터리 구조

```
commerce_agent/
├── main.py                      # FastAPI 진입점
├── graph.py                     # LangGraph StateGraph
├── state.py                     # AgentState TypedDict
├── nodes/
│   ├── intent.py                # classify_intent
│   ├── loaded.py                # check_loaded
│   ├── sql.py                   # run_sql
│   ├── ingestion.py             # run_ingestion (IngestionPipeline 실행)
│   └── responder.py             # generate_response
├── pipeline/
│   ├── base.py                  # Filter ABC, Pipeline
│   ├── kaggle_loader.py         # KaggleLoadFilter
│   ├── normalizer.py            # NormalizeFilter (규칙 기반, LLM 불필요)
│   └── db_writer.py             # UpsertFilter
├── db/
│   ├── models.py                # SQLAlchemy 모델
│   ├── session.py               # DB 세션
│   └── crud.py                  # upsert / 조회 함수
└── schemas.py                   # NormalizedProduct Pydantic 모델
```

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
    ├─ structured  →  [check_loaded]  ──loaded──→  [run_sql]  →  [generate_response]
    │                     └──not_loaded──→  [run_ingestion]  →  [run_sql]  →  [generate_response]
    └─ interpret   →  [check_loaded]  ──loaded──→  [run_sql]  →  [generate_response]
                            └──not_loaded──→  [run_ingestion]  →  [run_sql]  →  [generate_response]

[run_ingestion] = IngestionPipeline(KaggleLoad → Normalize → Upsert)
```

---

## LangGraph 워크플로우

### AgentState

그래프 전체를 흐르는 공유 상태. 모든 노드는 이 state를 읽고 업데이트한다.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    query:        str
    intent:       str | None          # "structured" | "interpret"
    is_loaded:    bool | None         # 해당 카테고리 데이터가 DB에 적재되어 있는가
    products:     list[dict]          # 적재된 상품 목록 (ingestion 후 채워짐)
    sql_query:    str | None          # 생성된 SQL
    sql_results:  list[dict]          # SQL 실행 결과
    sql_error:    str | None          # Stage 2 retry 실패 시 오류 메시지
    response:     str | None          # 최종 응답
    messages:     Annotated[list, add_messages]  # 대화 히스토리
```

### 노드 목록

| 노드 | 파일 | 역할 |
|---|---|---|
| `classify_intent` | `nodes/intent.py` | LLM으로 질의 유형 분류 |
| `check_loaded` | `nodes/loaded.py` | 해당 카테고리 데이터가 DB에 적재됐는지 확인 |
| `run_ingestion` | `nodes/ingestion.py` | IngestionPipeline 실행 |
| `run_sql` | `nodes/sql.py` | NL→SQL 변환 + 실행 |
| `generate_response` | `nodes/responder.py` | 최종 응답 생성 |

`normalize`, `upsert_db`는 LangGraph 노드가 아닌 `IngestionPipeline` 내부 Filter로 처리한다.

### 그래프 정의 (`graph.py`)

```python
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes.intent import make_classify_intent
from nodes.loaded import make_check_loaded
from nodes.ingestion import make_run_ingestion
from nodes.sql import make_run_sql
from nodes.responder import make_generate_response
from pipeline.base import Pipeline
from pipeline.kaggle_loader import KaggleLoadFilter
from pipeline.normalizer import NormalizeFilter
from pipeline.db_writer import UpsertFilter
from db.session import get_session
import kagglehub

def route_by_intent(state: AgentState) -> str:
    return state["intent"]  # "structured" | "interpret"

def route_by_loaded(state: AgentState) -> str:
    return "loaded" if state["is_loaded"] else "not_loaded"

def build_graph(dataset_path: str, session) -> StateGraph:
    pipeline = Pipeline([
        KaggleLoadFilter(dataset_path),
        NormalizeFilter(),
        UpsertFilter(session),
    ])

    g = StateGraph(AgentState)
    g.add_node("classify_intent",   make_classify_intent())
    g.add_node("check_loaded",      make_check_loaded(session))
    g.add_node("run_ingestion",     make_run_ingestion(pipeline))
    g.add_node("run_sql",           make_run_sql(session))
    g.add_node("generate_response", make_generate_response())

    g.set_entry_point("classify_intent")

    g.add_conditional_edges("classify_intent", route_by_intent, {
        "structured": "check_loaded",
        "interpret":  "check_loaded",
    })
    g.add_conditional_edges("check_loaded", route_by_loaded, {
        "loaded":     "run_sql",
        "not_loaded": "run_ingestion",
    })
    g.add_edge("run_ingestion",     "run_sql")
    g.add_edge("run_sql",           "generate_response")
    g.add_edge("generate_response", END)

    return g.compile()

# FastAPI lifespan에서 호출
# dataset_path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")
# commerce_graph = build_graph(dataset_path, get_session())
```

---


## 수집 파이프라인 — Pipe-Filter

데이터 소스로 **Amazon Products Sales Dataset 2023** (Kaggle 공개 데이터)을 사용한다.
웹 크롤링 없이 로컬 CSV를 읽어 DB에 적재하므로 Tavily, httpx, Playwright 의존이 불필요하다.

```python
import kagglehub
path = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")
```

파이프라인: `KaggleLoadFilter → NormalizeFilter → UpsertFilter`

### 데이터셋 스키마 → NormalizedProduct 매핑

| CSV 컬럼 | `NormalizedProduct` 필드 |
|---|---|
| `name` | `name` |
| `main_category` | `category` |
| `sub_category` | `facts["sub_category"]` |
| `ratings` | `rating` |
| `no_of_ratings` | `review_count` |
| `discount_price` | `price` |
| `actual_price` | `facts["actual_price"]` |
| `link` | `source_url` |
| `image` | `facts["image_url"]` |

`source_site = "amazon_kaggle_2023"` 고정.

### 구조 (`pipeline/`)

```python
# pipeline/base.py
from abc import ABC, abstractmethod
from typing import Any

class Filter(ABC):
    @abstractmethod
    async def process(self, data: Any) -> Any: ...

class Pipeline:
    def __init__(self, filters: list[Filter]):
        self.filters = filters

    async def run(self, data: Any) -> Any:
        for f in self.filters:
            data = await f.process(data)
        return data
```

### 각 Filter

```python
# pipeline/kaggle_loader.py
class KaggleLoadFilter(Filter):
    """카테고리명 str → list[dict] (CSV 행)"""
    def __init__(self, dataset_path: str): ...
    async def process(self, category: str) -> list[dict]:
        # pandas로 CSV 읽기
        # main_category == category 필터링
        # 결과를 dict 리스트로 반환
        ...

# pipeline/normalizer.py
class NormalizeFilter(Filter):
    """list[dict] (CSV 행) → list[NormalizedProduct]"""
    async def process(self, rows: list[dict]) -> list[NormalizedProduct]:
        # 가격 문자열 파싱 (₹1,234 → 1234)
        # 평점 문자열 파싱 ("4.2 out of 5" → 4.2)
        # NormalizedProduct로 변환
        # LLM 호출 없이 규칙 기반 파싱으로 충분
        ...

# pipeline/db_writer.py
class UpsertFilter(Filter):
    """list[NormalizedProduct] → list[NormalizedProduct]"""
    def __init__(self, session): ...
    async def process(self, products: list[NormalizedProduct]) -> list[NormalizedProduct]:
        # (source_site, source_url) 기준 upsert
        ...
```

> `NormalizeFilter`는 CSV 컬럼이 정형화되어 있으므로 **LLM 없이 규칙 기반 파싱**으로 구현한다.
> LLM structured output은 비정형 텍스트가 입력일 때만 필요하다.

### `run_ingestion` 노드 (`nodes/ingestion.py`)

```python
def make_run_ingestion(pipeline: Pipeline):
    async def run_ingestion(state: AgentState) -> AgentState:
        # query에서 카테고리 추출 후 파이프라인 실행
        category = state["query"]  # Phase 1: 질의를 카테고리로 직접 사용 (이어폰 단일 카테고리)
        products = await pipeline.run(category)
        return {**state, "products": [p.model_dump() for p in products]}
    return run_ingestion
```

### `check_loaded` 기준

데이터 소스는 Kaggle 정적 CSV이므로 TTL/신선도 개념은 적용하지 않는다.
**해당 카테고리 데이터가 DB에 적재되어 있는지** 여부만 판단한다.

```python
async def check_loaded(state: AgentState) -> AgentState:
    category = "headphones"  # Phase 1: 이어폰 단일 카테고리 고정
    count = await session.scalar(
        select(func.count()).where(normalized_products.c.category == category)
    )
    return {**state, "is_loaded": count > 0}
```

- `is_loaded = True` → 이미 적재됨, 바로 SQL 실행
- `is_loaded = False` → 미적재, `run_ingestion` 실행 후 SQL

### 파이프-필터의 이점

| 이점 | 설명 |
|---|---|
| 독립 테스트 | 각 Filter를 목(mock) 없이 단독으로 테스트 가능 |
| 단계 교체 | `KaggleLoadFilter`를 다른 데이터소스 Filter로 교체 가능 |
| 단계 추가 | `normalize` 전에 `DeduplicateFilter` 삽입 가능 |
| 단일 책임 | 각 Filter는 하나의 변환만 담당 |

---

## DB 스키마

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
| `interpret` | "가성비 좋은 러닝화 추천", "이 둘 차이가 뭐야?" |

> 데이터 소스가 Kaggle 정적 CSV이므로 실시간 재고/가격 조회(`freshness` intent)는 지원하지 않는다.

### `run_sql` (`nodes/sql.py`) — 3단계 점진적 전략

SQL 생성은 품질 지표에 따라 단계적으로 고도화한다.

```
Stage 1 (POC)     범용 LLM이 질의 → SQL 직접 생성
       ↓ 품질 지표가 목표 미달
Stage 2 (안정화)  Validator + Retry 루프 추가
       ↓ SQL 생성이 여전히 병목
Stage 3 (전문화)  범용 LLM은 의도 해석만, SQLCoder가 SQL 생성
```

#### Stage 1 — 직접 생성 (POC)

```python
async def run_sql(state: AgentState) -> AgentState:
    # 시스템 프롬프트에 DB 스키마 포함
    # LLM이 SELECT 문 생성
    # SELECT-only 검증 후 실행
```

#### SQL 품질 지표 측정

Stage 2 전환 여부는 `evals/` 골든셋 실행 결과로 판단한다.

| 지표 | 설명 | 측정 방법 |
|---|---|---|
| `schema_invalid_rate` | 존재하지 않는 테이블·컬럼 참조 비율 | 스키마 목록 대조 |
| `execution_failure_rate` | DB 실행 오류 발생 비율 | PostgreSQL 오류 캐치 |

목표 수준을 초과하면 Stage 2를 적용한다.

#### Stage 2 — Validator + Retry

```python
MAX_RETRIES = 3

async def run_sql(state: AgentState) -> AgentState:
    error_context = ""
    for attempt in range(MAX_RETRIES):
        sql = await llm.generate(SCHEMA_PROMPT + state["query"] + error_context)
        violation = validate(sql)
        if violation is None:
            results = await execute(sql)
            return {**state, "sql_query": sql, "sql_results": results}
        error_context = f"\n이전 시도 오류: {violation}\n수정하여 재작성하라."
    return {**state, "sql_results": [], "sql_error": "max retries exceeded"}
```

**Validator 검사 항목:**

| 검사 | 방법 |
|---|---|
| SELECT-only | `sqlparse`로 statement type 확인 |
| 테이블명 존재 | 스키마 테이블 목록과 대조 |
| 컬럼명 존재 | 스키마 컬럼 목록과 대조 |
| 실행 오류 | `EXPLAIN` 후 PostgreSQL 오류 캐치 |

#### Stage 3 — 범용 LLM + SQLCoder 분리 (SQL이 병목일 시)

범용 LLM은 자연어 의도를 구조화하고, SQL 전용 모델이 스키마 기반 SQL을 생성한다.

```python
async def run_sql(state: AgentState) -> AgentState:
    # Step 1: 범용 LLM — 의도를 구조화된 형태로 파싱
    intent = await general_llm.with_structured_output(QueryIntent).invoke(state["query"])
    # 예: {"filter": {"price_lte": 50000, "category": "이어폰"}, "sort": "rating desc"}

    # Step 2: SQLCoder — 스키마 + 구조화된 의도 → SQL 생성
    sql = await sqlcoder_llm.invoke(SCHEMA_PROMPT + str(intent))

    results = await execute(sql)
    return {**state, "sql_query": sql, "sql_results": results}
```

**SQLCoder 모델:**

| 환경 | 범용 LLM | SQLCoder |
|---|---|---|
| Ollama (POC) | `llama3.1:8b` | `sqlcoder:7b` |
| HuggingFace (Phase 2+) | `meta-llama/Llama-3.1-8B-Instruct` | `defog/sqlcoder-7b-2` |

### `NormalizedProduct` 스키마 (`schemas.py`)

```python
class NormalizedProduct(BaseModel):
    source_site:   str
    source_url:    str
    name:          str
    brand:         str | None
    category:      str | None
    price:         int | None
    currency:      str = "INR"   # Amazon Kaggle 데이터셋 기준
    rating:        float | None
    review_count:  int | None
    facts:         dict[str, str] = {}
```

`NormalizeFilter`는 Kaggle CSV 컬럼을 규칙 기반으로 파싱한다. LLM 호출 없음.

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

## 테스트 전략

### 테스트 레벨

```
          ┌─────────────────┐
          │  통합 테스트     │  전체 파이프라인, 실제 DB, 실제 LLM
          ├─────────────────┤
          │  계약 테스트     │  LLM 노드 — 출력 구조 검증
          ├─────────────────┤
          │  단위 테스트     │  Filter, check_loaded, 라우팅 — TDD
          └─────────────────┘
```

### 단위 테스트 — TDD 적용 (Filter / 순수 로직)

Filter는 입출력이 명확한 순수 변환이라 TDD 사이클이 자연스럽게 맞는다.
Red(실패 테스트 작성) → Green(최소 구현) → Refactor 순서로 진행한다.

```python
# tests/pipeline/test_normalizer.py
import pytest
from pipeline.normalizer import NormalizeFilter

@pytest.mark.asyncio
async def test_extracts_price():
    raw = [{"text": "나이키 에어맥스 99,000원 ★4.5 리뷰 312개", "url": "http://...", "site": "musinsa"}]
    result = await NormalizeFilter().process(raw)
    assert result[0].price == 99000
    assert result[0].rating == 4.5
    assert result[0].review_count == 312

@pytest.mark.asyncio
async def test_missing_price_is_none():
    raw = [{"text": "품절된 상품입니다", "url": "http://...", "site": "musinsa"}]
    result = await NormalizeFilter().process(raw)
    assert result[0].price is None
```

```python
# tests/nodes/test_loaded.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_check_loaded_when_data_exists():
    with patch("nodes.loaded.get_category_count", return_value=42):
        from nodes.loaded import make_check_loaded
        node = make_check_loaded(session=AsyncMock())
        result = await node({"query": "이어폰 추천", "messages": []})
    assert result["is_loaded"] is True

@pytest.mark.asyncio
async def test_check_loaded_when_empty():
    with patch("nodes.loaded.get_category_count", return_value=0):
        from nodes.loaded import make_check_loaded
        node = make_check_loaded(session=AsyncMock())
        result = await node({"query": "이어폰 추천", "messages": []})
    assert result["is_loaded"] is False
```

```python
# tests/test_routing.py
from graph import route_by_intent, route_by_loaded

def test_route_structured():
    assert route_by_intent({"intent": "structured"}) == "structured"

def test_route_loaded():
    assert route_by_loaded({"is_loaded": True}) == "loaded"

def test_route_not_loaded():
    assert route_by_loaded({"is_loaded": False}) == "not_loaded"
```

### 계약 테스트 — LLM 노드

LLM 출력은 비결정적이므로 내용이 아닌 **구조와 타입**만 검증한다.

```python
# tests/nodes/test_intent.py
import pytest
from nodes.intent import make_classify_intent

VALID_INTENTS = {"structured", "interpret"}

@pytest.mark.asyncio
async def test_classify_intent_returns_valid_type():
    node = make_classify_intent()
    result = await node({"query": "5만원 이하 이어폰", "messages": []})
    assert result["intent"] in VALID_INTENTS

@pytest.mark.asyncio
async def test_classify_structured_query():
    node = make_classify_intent()
    result = await node({"query": "나이키 러닝화 평점 높은 순", "messages": []})
    assert result["intent"] == "structured"  # 명백한 케이스만 내용 검증
```

### Mock 단위 테스트 — 외부 의존

`KaggleLoadFilter`는 `kagglehub.dataset_download`를 Mock으로 격리해 CSV 파싱 로직만 테스트한다.

```python
# tests/pipeline/test_kaggle_loader.py
from unittest.mock import patch
from pipeline.kaggle_loader import KaggleLoadFilter

@pytest.mark.asyncio
async def test_kaggle_loader_filters_by_category(tmp_path):
    # 테스트용 미니 CSV 생성
    csv_content = "name,main_category,discount_price,ratings,no_of_ratings,link\n"
    csv_content += "Sony WH-1000XM5,headphones,₹29,990,4.3,1234,http://...\n"
    csv_content += "Nike Air Max,shoes,₹8,999,4.5,567,http://...\n"
    csv_file = tmp_path / "headphones.csv"
    csv_file.write_text(csv_content)

    with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
        result = await KaggleLoadFilter(str(tmp_path)).process("headphones")
    assert len(result) == 1
    assert result[0]["main_category"] == "headphones"
```

### 통합 테스트 — 전체 파이프라인

실제 DB(테스트용)를 사용해 파이프라인 전체를 검증한다.
Kaggle 데이터 다운로드가 필요하므로 CI에서는 선택적으로 실행한다(`-m integration` 마커).

```python
# tests/integration/test_pipeline.py
import pytest
from pipeline.base import Pipeline
from pipeline.kaggle_loader import KaggleLoadFilter
from pipeline.normalizer import NormalizeFilter
from pipeline.db_writer import UpsertFilter

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_ingestion_pipeline(test_db_session, kaggle_dataset_path):
    pipeline = Pipeline([
        KaggleLoadFilter(kaggle_dataset_path),
        NormalizeFilter(),
        UpsertFilter(test_db_session),
    ])
    products = await pipeline.run("headphones")
    assert len(products) > 0
    assert all(p.name for p in products)
```

### pytest 설정 (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "integration: 외부 API, 실제 DB 사용 (CI 선택적 실행)",
]

[tool.pytest.ini_options.filterwarnings]
# SQLAlchemy async 경고 숨김
ignore::DeprecationWarning
```

### 테스트 디렉터리

```
tests/
├── pipeline/
│   ├── test_kaggle_loader.py # KaggleLoadFilter (Mock)
│   ├── test_normalizer.py    # NormalizeFilter — TDD
│   └── test_db_writer.py     # UpsertFilter (테스트 DB)
├── nodes/
│   ├── test_intent.py        # 계약 테스트
│   └── test_loaded.py        # check_loaded 순수 로직 — TDD
├── test_routing.py           # route_by_* 함수 — TDD
└── integration/
    └── test_pipeline.py      # 전체 파이프라인
```

### 평가(Evaluation) 레이어 — 골든셋

계약 테스트가 출력 **구조**를 검증한다면, 평가 레이어는 출력 **내용**의 품질을 검증한다.
이 시스템의 핵심은 SQL 기반 데이터 조회이므로, 모든 LLM 노드 품질은 골든셋으로 검증 가능하다.

#### `classify_intent` — 골든셋

```python
# evals/test_intent_golden.py
GOLDEN = [
    ("5만원 이하 이어폰",          "structured"),
    ("나이키 평점 높은 순",         "structured"),
    ("가성비 좋은 키보드 추천해줘",  "interpret"),
    ("이 둘 차이가 뭐야?",          "interpret"),
]

@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected", GOLDEN)
async def test_intent_golden(query, expected):
    node = make_classify_intent()
    result = await node({"query": query, "messages": []})
    assert result["intent"] == expected
```

#### `NormalizeFilter` — 골든셋

```python
# evals/test_normalize_golden.py
GOLDEN = [
    (
        "나이키 에어맥스 99,000원 ★4.5 리뷰 312개",
        {"brand": "나이키", "price": 99000, "rating": 4.5, "review_count": 312},
    ),
    (
        "소니 WH-1000XM5 349,000원 품절",
        {"brand": "소니", "price": 349000},
    ),
]

@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("text, expected_fields", GOLDEN)
async def test_normalize_golden(text, expected_fields):
    products = await NormalizeFilter().process([{"text": text, "url": "...", "site": "test"}])
    product = products[0].model_dump()
    for field, value in expected_fields.items():
        assert product[field] == value
```

#### `run_sql` — 테스트 DB + 결과 검증

SQL 문자열을 직접 비교하면 의미가 같아도 형태가 달라 깨진다.
테스트 DB에 고정 데이터를 넣고 **실행 결과**로 검증한다.

```python
# evals/test_sql_golden.py
GOLDEN = [
    (
        "5만원 이하 이어폰",
        {"min_count": 1, "max_price": 50000, "category": "이어폰"},
    ),
    (
        "나이키 신발 평점 높은 순 3개",
        {"count": 3, "brand": "나이키", "ordered_by_rating": True},
    ),
]

@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("query, constraint", GOLDEN)
async def test_sql_results(query, constraint, test_db_session):
    node = make_run_sql(test_db_session)
    result = await node({"query": query, "messages": [], "intent": "structured"})
    rows = result["sql_results"]

    if "max_price" in constraint:
        assert all(r["price"] <= constraint["max_price"] for r in rows)
    if "category" in constraint:
        assert all(r["category"] == constraint["category"] for r in rows)
    if "count" in constraint:
        assert len(rows) == constraint["count"]
    if "min_count" in constraint:
        assert len(rows) >= constraint["min_count"]
    if constraint.get("ordered_by_rating"):
        ratings = [r["rating"] for r in rows]
        assert ratings == sorted(ratings, reverse=True)
```

#### `generate_response` — SQL 결과 포함 여부

`generate_response`는 SQL 결과를 텍스트로 포장하는 역할이다.
SQL 결과의 상품명·가격이 응답에 포함되어 있는지 단순 검증으로 충분하다.

```python
# tests/nodes/test_responder.py
@pytest.mark.asyncio
async def test_response_includes_product_names():
    sql_results = [
        {"name": "소니 WH-1000XM5", "price": 349000},
        {"name": "애플 에어팟 프로", "price": 359000},
    ]
    state = await make_generate_response()({"sql_results": sql_results, "query": "...", "messages": []})
    assert "소니 WH-1000XM5" in state["response"]
    assert "애플 에어팟 프로" in state["response"]
```

#### 실행 분리

```toml
[tool.pytest.ini_options]
markers = [
    "integration: 외부 API, 실제 DB 사용",
    "eval: 골든셋 평가 (LLM 호출, 수동 실행)",
]
```

```bash
uv run pytest -m "not integration and not eval"   # CI
uv run pytest -m eval                             # 프롬프트 변경 후 수동
```

---

### 요약

| 대상 | 방법 | 실행 시점 |
|---|---|---|
| Filter 변환 로직 | 단위 테스트 (TDD) | 매 커밋 |
| `check_loaded` 로직 | 단위 테스트 (TDD) | 매 커밋 |
| 라우팅 함수 | 단위 테스트 (TDD) | 매 커밋 |
| LLM 노드 구조 | 계약 테스트 | 매 커밋 |
| Kaggle 로더 | Mock 단위 테스트 | 매 커밋 |
| 전체 파이프라인 | 통합 테스트 (`-m integration`) | PR / 수동 |
| `classify_intent` 품질 | 골든셋 (`-m eval`) | 프롬프트 변경 후 |
| `NormalizeFilter` 품질 | 골든셋 (`-m eval`) | 규칙 변경 후 |
| `run_sql` 품질 | 테스트 DB + 결과 검증 (`-m eval`) | 프롬프트 변경 후 |
| `generate_response` 품질 | SQL 결과 포함 여부 단순 검증 | 매 커밋 |

---

## 구현 순서 (Phase)

### Phase 1 — KISS 구현 (현재 목표)

> 목표: 이어폰 단일 카테고리 end-to-end 동작 확인
> 적용: FastAPI + LangGraph + IngestionPipeline + SQLAlchemy 직접 호출

- [ ] SQLAlchemy 모델 + Alembic 마이그레이션 (`normalized_products`, `product_facts`)
- [ ] `Pipeline` + `Filter` ABC (`pipeline/base.py`)
- [ ] `KaggleLoadFilter` — `kagglehub`으로 CSV 다운로드 + pandas 로드
- [ ] `NormalizeFilter` — 규칙 기반 CSV 컬럼 파싱 → `NormalizedProduct` (LLM 불필요)
- [ ] `UpsertFilter` — SQLAlchemy upsert (`source_site`, `source_url` 기준)
- [ ] `IngestionPipeline` end-to-end 단독 테스트
- [ ] LangGraph `StateGraph` 기본 골격 + 노드 팩토리 패턴 적용
- [ ] `classify_intent` + `check_loaded` + `run_ingestion` + `run_sql` + `generate_response`
- [ ] FastAPI `POST /query` 엔드포인트 연결
- [ ] `check_loaded` 판단 기준: 해당 카테고리 상품이 DB에 1건 이상 존재하면 loaded

### Phase 2 — 구조 개선

> 목표: Clean Architecture 레이어 분리, 커버리지 확대
> 적용: Repository 인터페이스, DI, 멀티턴 대화

- [ ] `IProductRepository` 인터페이스 도입 + 노드 팩토리에 주입
- [ ] `interface/`, `application/`, `domain/`, `infrastructure/` 레이어 분리
- [ ] LangGraph `MemorySaver` — 멀티턴 대화 상태 유지
- [ ] 커버리지 확대 (무신사, 올리브영)
- [ ] `GET /products`, `POST /refresh/{id}` 엔드포인트

### Phase 3 — 품질 개선

- [ ] DDD Aggregate / Value Object / Domain Event 도입
- [ ] Bounded Context 분리 (Product Catalog / Ingestion / Query)
- [ ] Redis 캐시 (자주 조회되는 SQL 결과)
- [ ] pgvector + RAG (`interpret` 경로)
- [ ] `source_provenance` 테이블

---

## DDD (도메인 주도 설계)

### 서브도메인 분류

| 서브도메인 | 유형 | 설명 |
|---|---|---|
| **Product Catalog** | Core Domain | 상품 정보를 구조화·축적하는 핵심 차별화 영역 |
| **Ingestion** | Supporting Domain | 웹 수집 → 정규화 → DB 적재 파이프라인 |
| **Query** | Supporting Domain | 질의 분류, SQL 생성, 응답 생성 |
| **Provenance** | Generic Domain | 출처 추적, 신뢰도 관리 |

---

### Bounded Context

```
┌─────────────────────────┐   ┌─────────────────────────┐
│   Product Catalog BC    │   │      Ingestion BC        │
│                         │   │                          │
│  Aggregate: Product     │◄──│  Domain Svc:             │
│  - ProductFact (VO)     │   │    NormalizationService  │
│  - Price (VO)           │   │  Aggregate: SearchResult │
│  - Rating (VO)          │   │  - SourceUrl (VO)        │
│  Repo: IProductRepo     │   │  - Confidence (VO)       │
│  Event: ProductUpserted │   │  Repo: ISearchResultRepo │
└─────────────────────────┘   └─────────────────────────┘
           ▲                              ▲
           │         Anti-Corruption Layer│
           └──────────────┬───────────────┘
                          │
            ┌─────────────────────────┐
            │        Query BC         │
            │                         │
            │  App Svc: QueryService  │
            │  Domain Svc:            │
            │    IntentClassifier     │
            │    SqlGenerator         │
            │    FreshnessChecker     │
            └─────────────────────────┘
```

Context Map 관계:
- **Ingestion → Product Catalog**: Ingestion이 수집한 데이터를 Product Catalog에 `ProductUpserted` 이벤트로 전달. ACL(Anti-Corruption Layer)에서 외부 웹 데이터 모델을 Product 도메인 모델로 변환.
- **Query → Product Catalog**: Query BC는 Product Catalog를 읽기 전용으로 사용 (Customer/Supplier).

---

### 레이어 구조

```
┌──────────────────────────────────┐
│         Interface Layer          │  FastAPI 라우터, 요청/응답 직렬화
├──────────────────────────────────┤
│        Application Layer         │  LangGraph StateGraph, 유스케이스 조율
├──────────────────────────────────┤
│          Domain Layer            │  Aggregate, Entity, VO, Domain Service,
│                                  │  Repository 인터페이스, Domain Event
├──────────────────────────────────┤
│       Infrastructure Layer       │  SQLAlchemy, Tavily, httpx, Anthropic SDK
└──────────────────────────────────┘
```

**의존성 방향**: Interface → Application → Domain ← Infrastructure
Domain Layer는 외부 라이브러리에 의존하지 않는다.

---

### Domain Model

#### Product Aggregate (`domain/product/`)

```python
# entity.py
@dataclass
class Product:                        # Aggregate Root
    id:           ProductId           # VO
    source:       ProductSource       # VO (site + url)
    name:         str
    brand:        str | None
    category:     str | None
    price:        Price | None        # VO
    rating:       Rating | None       # VO
    facts:        list[ProductFact]   # VO list
    updated_at:   datetime

    def is_stale(self, ttl: TTLPolicy) -> bool: ...
    def apply_update(self, data: ProductData) -> None: ...  # 상태 변경은 메서드로만
```

```python
# value_objects.py
@dataclass(frozen=True)
class Price:
    amount:   int
    currency: str = "KRW"

@dataclass(frozen=True)
class Rating:
    score:        float   # 0.0 ~ 5.0
    review_count: int

    def __post_init__(self):
        if not 0.0 <= self.score <= 5.0:
            raise ValueError("rating must be between 0 and 5")

@dataclass(frozen=True)
class ProductFact:
    name:  str
    value: str

@dataclass(frozen=True)
class ProductSource:
    site: str
    url:  str
```

```python
# repository.py  — 인터페이스만 정의, 구현은 Infrastructure
from abc import ABC, abstractmethod

class IProductRepository(ABC):
    @abstractmethod
    async def find_by_source(self, source: ProductSource) -> Product | None: ...
    @abstractmethod
    async def save(self, product: Product) -> None: ...
    @abstractmethod
    async def find_by_filter(self, f: ProductFilter) -> list[Product]: ...
```

```python
# events.py
@dataclass
class ProductUpserted:
    product_id: ProductId
    source:     ProductSource
    occurred_at: datetime
```

#### SearchResult Aggregate (`domain/ingestion/`)

```python
# entity.py
@dataclass
class SearchResult:                   # Aggregate Root
    id:         SearchResultId
    query:      str
    title:      str | None
    snippet:    str | None
    url:        str
    source:     str
    raw_text:   str | None
    fetched_at: datetime
```

#### Domain Services

```python
# domain/ingestion/services.py
class NormalizationService:
    """비정형 웹 텍스트 → Product 도메인 모델 변환. LLM 호출은 포트(인터페이스)로 분리."""
    def __init__(self, llm_port: ILLMPort): ...
    async def normalize(self, result: SearchResult) -> Product: ...

# domain/product/services.py
class FreshnessService:
    """TTL 정책에 따라 Product의 stale 여부 판단."""
    def __init__(self, policy: TTLPolicy): ...
    def is_stale(self, product: Product, field: str) -> bool: ...
```

---

### 디렉터리 구조 (DDD)

```
commerce_agent/
├── main.py
│
├── interface/                        # Interface Layer
│   └── api/
│       ├── routes.py                 # FastAPI 라우터
│       └── schemas.py                # 요청/응답 Pydantic 모델
│
├── application/                      # Application Layer
│   ├── graph.py                      # LangGraph StateGraph 조립
│   ├── state.py                      # AgentState TypedDict
│   └── nodes/
│       ├── intent.py                 # classify_intent
│       ├── loaded.py                 # check_loaded
│       ├── sql.py                    # run_sql
│       ├── search.py                 # web_search
│       ├── crawler.py                # crawl
│       ├── normalizer.py             # normalize (NormalizationService 호출)
│       ├── db_writer.py              # upsert_db (IProductRepository 호출)
│       └── responder.py              # generate_response
│
├── domain/                           # Domain Layer (외부 의존 없음)
│   ├── product/
│   │   ├── entity.py                 # Product aggregate root
│   │   ├── value_objects.py          # Price, Rating, ProductFact, ProductSource
│   │   ├── repository.py             # IProductRepository (인터페이스)
│   │   ├── events.py                 # ProductUpserted
│   │   └── services.py               # FreshnessService, TTLPolicy
│   └── ingestion/
│       ├── entity.py                 # SearchResult aggregate
│       ├── value_objects.py          # Confidence, SourceUrl
│       ├── repository.py             # ISearchResultRepository (인터페이스)
│       └── services.py               # NormalizationService
│
└── infrastructure/                   # Infrastructure Layer
    ├── db/
    │   ├── models.py                 # SQLAlchemy ORM 모델
    │   ├── session.py                # DB 세션
    │   └── repositories/
    │       ├── product_repo.py       # IProductRepository 구현체
    │       └── search_repo.py        # ISearchResultRepository 구현체
    ├── search/
    │   └── tavily_client.py          # Tavily API 클라이언트
    ├── crawler/
    │   └── httpx_crawler.py          # httpx + BeautifulSoup4
    └── llm/
        ├── anthropic_client.py       # Anthropic SDK 래퍼
        └── ports.py                  # ILLMPort 인터페이스
```

---

## 아키텍처 패턴 / 디자인 패턴

### 아키텍처 패턴

**Cache-Aside (Lazy Population)**
DB를 캐시처럼 운용한다. 질의가 들어오면 DB를 먼저 확인하고, miss이거나 stale이면 원본(웹)에서 가져와 채운다. 애플리케이션이 캐시 채우기를 직접 책임지는 Cache-Aside 패턴과 동일한 구조.

```
read(query) → DB hit? → return
                ↓ miss/stale
             web search → normalize → upsert → return
```

**CQRS (Command Query Responsibility Segregation)**
읽기와 쓰기 경로를 분리한다.
- **Query 경로**: `run_sql` 노드 → `normalized_products` 조회 (SQL Agent)
- **Command 경로**: `web_search → crawl → normalize → upsert_db` (수집 파이프라인)

두 경로가 동일 모델을 공유하되 코드 흐름은 완전히 분리되어 있다.

**Pipeline (ETL)**
수집 경로는 전형적인 ETL 파이프라인이다.

| 단계 | 노드 | 역할 |
|---|---|---|
| Extract | `web_search` + `crawl` | 원천 데이터 수집 |
| Transform | `normalize` | 비정형 → `NormalizedProduct` 구조화 |
| Load | `upsert_db` | DB 적재 |

**Layered Architecture**
```
API Layer          FastAPI 엔드포인트
Agent Layer        LangGraph StateGraph (intent → routing → 실행)
Service Layer      각 노드 함수 (search, crawl, normalize, sql)
DB Layer           SQLAlchemy models + crud
```

---

### 디자인 패턴

**Strategy**
`classify_intent` 노드가 intent에 따라 실행 전략을 선택한다. `structured`, `interpret` 두 전략은 같은 인터페이스(`AgentState → AgentState`)를 공유하지만 내부 경로가 다르다.

**Repository**
`IProductRepository` 인터페이스를 Domain Layer에 정의하고 구현체를 Infrastructure Layer에 둔다. Application Layer(노드)는 인터페이스에만 의존하므로 DB 교체 시 노드 코드를 건드리지 않아도 된다.

**EAV (Entity-Attribute-Value)**
`product_facts` 테이블이 EAV 패턴. 카테고리마다 다른 속성(이어폰의 "드라이버 크기", 신발의 "갑피 소재")을 스키마 변경 없이 저장한다. 도메인 모델에서는 `list[ProductFact]`(VO)로 표현되며 영속화 시에만 EAV 구조로 매핑된다. 단, 조회 복잡도 증가 트레이드오프가 있다.

**Facade**
FastAPI의 `POST /query` 엔드포인트가 Facade. 클라이언트는 LangGraph 그래프 내부 노드 구성이나 DB 구조를 몰라도 된다. `commerce_graph.ainvoke()` 한 번으로 전체 파이프라인이 실행된다.

**Strangler Fig (점진적 교체)**
Phase 1에서는 단일 카테고리(이어폰)로 파이프라인을 검증하고, Phase 3에서 커버리지를 확대한다. 초기에는 웹검색 의존도가 높다가 DB가 채워질수록 검색 비율이 줄어드는 구조 자체가 Strangler Fig와 같은 점진적 전환이다.

---

## 기술 스택

| 역할 | POC (Phase 1) | Phase 2+ |
|---|---|---|
| API | FastAPI + uvicorn | 동일 |
| 에이전트 워크플로우 | LangGraph | 동일 |
| DB | PostgreSQL + SQLAlchemy (async) + Alembic | 동일 |
| **범용 LLM** | **Ollama (`llama3.1:8b`)** | **HuggingFace (`meta-llama/Llama-3.1-8B-Instruct`)** |
| **SQL LLM (Stage 3)** | **Ollama (`sqlcoder:7b`)** | **HuggingFace (`defog/sqlcoder-7b-2`)** |
| **LLM 서빙** | **Ollama** | **SGLang 또는 vLLM (벤치마크로 결정)** |
| **LLM 클라이언트** | **`langchain-ollama`** | **`langchain-openai` (OpenAI-compatible)** |
| **데이터 소스** | **Kaggle (`kagglehub`, 정적 CSV)** | 동일 |
| 캐시 (Phase 3) | — | Redis |
| 벡터 (Phase 3) | — | pgvector |
| 테스트 | pytest + pytest-asyncio | 동일 |

### LLM 전환 전략

POC에서는 Ollama로 로컬 실행해 빠르게 검증한다.
Phase 2부터 `langchain-ollama`를 제거하고 `meta-llama/Llama-3.1-8B-Instruct`를 HuggingFace에서 다운로드해 **SGLang 또는 vLLM**으로 GPU 서빙한다.
두 프레임워크 모두 OpenAI-compatible API를 노출하므로 결정 전/후 모두 `ChatOpenAI`(`base_url` 교체)로 접근한다.

```python
# POC (Phase 1) — langchain-ollama
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")

# Phase 2+ — SGLang 또는 vLLM (OpenAI-compatible)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:30000/v1",  # SGLang 기본 포트 (vLLM: 8000)
    api_key="EMPTY",
)
```

노드 팩토리에 `llm`을 주입하면 나머지 코드는 변경 없다.

```python
# graph.py
llm = ChatOllama(...)    # Phase 1
# llm = ChatOpenAI(...) # Phase 2+ (SGLang / vLLM 결정 후)
g.add_node("classify_intent", make_classify_intent(llm))
```

---

### Phase 2 — GPU 서빙 프레임워크 성능 테스트

SGLang과 vLLM을 동일 환경에서 벤치마크해 프레임워크를 결정한다.

#### 서버 실행

```bash
# SGLang
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000

# vLLM
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

#### 측정 지표

| 지표 | 설명 |
|---|---|
| TTFT (Time To First Token) | 첫 토큰까지의 지연 — 사용자 체감 응답속도 |
| TBT (Time Between Tokens) | 스트리밍 토큰 간격 |
| Throughput | 초당 생성 토큰 수 (tokens/sec) |
| Concurrency | 동시 요청 처리 성능 |
| VRAM | GPU 메모리 사용량 |

#### 테스트 시나리오

이 프로젝트의 실제 호출 패턴에 맞춘 프롬프트 유형으로 측정한다.

| 시나리오 | 대상 노드 | 프롬프트 길이 |
|---|---|---|
| Intent 분류 | `classify_intent` | 짧음 (< 200 tokens) |
| SQL 생성 | `run_sql` | 중간 (스키마 포함, ~800 tokens) |
| 상품 정규화 | `NormalizeFilter` | 김 (HTML 포함, ~3000 tokens) |
| 동시 요청 (×10) | 전체 | 혼합 |

#### 벤치마크 스크립트 (`benchmarks/run.py`)

```python
import asyncio, time, httpx, statistics

SCENARIOS = {
    "intent":      "다음 질의의 의도를 분류하라: 5만원 이하 이어폰",
    "sql":         "스키마: ...\n질의: 5만원 이하 이어폰을 평점 높은 순으로 조회하는 SQL을 작성하라.",
    "normalize":   "다음 HTML에서 상품 정보를 추출하라: <html>...</html>",
}

async def measure(base_url: str, prompt: str, n: int = 20) -> dict:
    latencies = []
    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(n):
            start = time.perf_counter()
            await client.post(f"{base_url}/v1/chat/completions", json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
            })
            latencies.append(time.perf_counter() - start)
    return {
        "mean":   round(statistics.mean(latencies), 3),
        "p50":    round(statistics.median(latencies), 3),
        "p95":    round(sorted(latencies)[int(n * 0.95)], 3),
    }

async def main():
    for name, url in [("SGLang", "http://localhost:30000"), ("vLLM", "http://localhost:8000")]:
        print(f"\n=== {name} ===")
        for scenario, prompt in SCENARIOS.items():
            result = await measure(url, prompt)
            print(f"  {scenario}: {result}")

asyncio.run(main())
```

#### 결정 기준

- **TTFT / p95 latency** 우선: 사용자 질의 응답이 주 목적
- **SQL 생성 시나리오** 가중: `run_sql`은 매 질의마다 호출되는 핵심 경로
- 동점 시 VRAM 사용량이 적은 쪽 선택
