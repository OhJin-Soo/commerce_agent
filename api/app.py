from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_ollama import ChatOllama

from agent import GraphDeps, build_graph
from api.routes import router
from db.session import AsyncSessionLocal
from pipeline.kaggle_load import KaggleDatasetConfig

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 LangGraph 를 한 번만 빌드해 app.state 에 보관한다."""
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ingest_handle = os.getenv(
        "KAGGLE_DATASET_HANDLE",
        "asaniczka/amazon-products-dataset-2023-1-4m-products",
    )
    ingest_nrows = int(os.getenv("KAGGLE_NROWS", "50000"))

    logger.info("Building graph  model=%s  dataset=%s", model, ingest_handle)

    deps = GraphDeps(
        session_factory=AsyncSessionLocal,
        llm=ChatOllama(model=model),
        ingest_config=KaggleDatasetConfig(
            handle=ingest_handle,
            nrows=ingest_nrows,
        ),
    )
    app.state.graph = build_graph(deps)
    logger.info("Graph ready")

    yield
    # 종료 시 정리할 자원 없음


def create_app() -> FastAPI:
    app = FastAPI(
        title="Commerce Agent",
        version="0.1.0",
        description="Search-augmented commerce agent API",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
