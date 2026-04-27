# commerce-agent

FastAPI + LangGraph 기반 커머스 에이전트입니다. Kaggle 상품 CSV를 PostgreSQL에 미리 적재한 뒤, API 요청에서는 적재 없이 DB 조회와 LLM 응답 생성만 수행합니다.

## Runtime

- Backend: FastAPI, LangGraph, SQLAlchemy async
- DB: PostgreSQL
- LLM: Ollama
- Frontend: Vite/React (`frontend/`)
- Optional: Tavily web search, LangSmith tracing

현재 API 운영 라우팅은 다음 기준입니다.

- 일반 DB 기반 상품 조회: `llama3.1:8b` + pipeline
- 리뷰/후기/평가 등 web search 수요: `gemma4:26b` + ReAct

## Docker 실행

환경파일을 준비합니다.

```bash
cp .env.example .env
```

필요하면 `.env`에서 모델과 외부 API 키를 조정합니다.

```env
OLLAMA_MODELS=llama3.1:8b,gemma4:26b
PIPELINE_MODEL=llama3.1:8b
WEB_SEARCH_MODEL=gemma4:26b
OLLAMA_HOST=http://host.docker.internal:11434
TAVILY_API_KEY=
LANGSMITH_API_KEY=
```

Docker로 API를 띄워도 Ollama 서버는 현재처럼 호스트에서 실행한다고 가정합니다. 따라서 로컬에 필요한 모델이 있어야 합니다.

```bash
ollama pull llama3.1:8b
ollama pull gemma4:26b
```

컨테이너를 빌드하고 실행합니다.

```bash
docker compose up --build
```

백그라운드 실행은 다음 명령을 사용합니다.

```bash
docker compose up -d --build
```

API는 기본적으로 아래 주소에서 뜹니다.

```text
http://localhost:8000
```

FastAPI Swagger UI는 아래 주소에서 확인하고 직접 테스트할 수 있습니다.

```text
http://localhost:8000/docs
```

브라우저에서 `http://localhost:8000/docs`를 열고 `POST /query`를 실행하면 됩니다.

## DB 마이그레이션

DB 컨테이너가 준비된 뒤 Alembic migration을 실행합니다.

```bash
docker compose exec api uv run alembic upgrade head
```

로컬 venv에서 실행하려면:

```bash
uv run alembic upgrade head
```

## 상품 데이터 Preload

API 요청 중에는 ingestion을 수행하지 않습니다. 상품 데이터는 별도 스크립트로 미리 적재합니다.

```bash
docker compose exec api uv run python preload.py
```

로컬에서 실행하려면:

```bash
uv run python preload.py
```

## API 사용 예시

일반 DB 조회는 `llama3.1:8b + pipeline` 경로로 자동 라우팅됩니다.

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"이어폰 5만원 이하"}'
```

리뷰/후기/평가 질의는 `gemma4:26b + ReAct` 경로로 자동 라우팅됩니다.

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Sony WH-1000XM5 리뷰"}'
```

응답의 `model` 필드는 실제 선택된 모델을 반환합니다.

## 로컬 개발

DB만 Docker로 띄우고 앱은 로컬에서 실행할 수 있습니다.

```bash
docker compose up -d db
uv run alembic upgrade head
uv run python preload.py
uv run python main.py
```

테스트:

```bash
uv run pytest
```

모델 평가:

```bash
uv run python eval_models.py --path all
```
