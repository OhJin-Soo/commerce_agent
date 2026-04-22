from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama

from agent import GraphDeps, build_graph
from api.routes import router
from db.currency import fetch_inr_to_krw
from db.session import AsyncSessionLocal

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 LangGraph 를 모델별로 한 번씩 빌드해 app.state 에 보관한다.

    환경 변수:
        OLLAMA_MODELS   쉼표 구분 모델 목록 (기본: "llama3.1:8b")
                        예) "llama3.1:8b,deepseek-r1:8b"
        KAGGLE_DATASET_HANDLE   Kaggle 데이터셋 핸들
        KAGGLE_NROWS            행 수 제한
    """
    models_env = os.getenv("OLLAMA_MODELS", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    model_names = [m.strip() for m in models_env.split(",") if m.strip()]
    default_model = model_names[0]

    ingest_handle = os.getenv(
        "KAGGLE_DATASET_HANDLE",
        "lokeshparab/amazon-products-dataset",
    )
    ingest_nrows = int(os.getenv("KAGGLE_NROWS", "50000"))
    tavily_api_key = os.getenv("TAVILY_API_KEY") or None

    exchange_rate = await fetch_inr_to_krw()
    logger.info(
        "Building graphs  models=%s  dataset=%s  INR→KRW=%.2f",
        model_names, ingest_handle, exchange_rate,
    )

    graphs: dict = {}
    for model in model_names:
        deps = GraphDeps(
            session_factory=AsyncSessionLocal,
            llm=ChatOllama(model=model),
            dataset_handle=ingest_handle,
            ingest_nrows=ingest_nrows,
            exchange_rate=exchange_rate,
            tavily_api_key=tavily_api_key,
        )
        graphs[model] = build_graph(deps)
        logger.info("Graph ready  model=%s", model)

    app.state.graphs = graphs
    app.state.default_model = default_model
    app.state.exchange_rate = exchange_rate

    yield
    # 종료 시 정리할 자원 없음


def create_app() -> FastAPI:
    app = FastAPI(
        title="Commerce Agent",
        version="0.1.0",
        description="Search-augmented commerce agent API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()
