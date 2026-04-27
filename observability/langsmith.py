from __future__ import annotations

import os
from typing import Any


DEFAULT_LANGSMITH_PROJECT = "commerce-agent"


def configure_langsmith() -> bool:
    """LANGSMITH_API_KEY 가 있으면 tracing 기본값을 활성화한다."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT)
    return True


def langsmith_enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY")) and (
        os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    )


def build_run_config(
    run_name: str,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not langsmith_enabled():
        return None
    return {
        "run_name": run_name,
        "tags": tags or [],
        "metadata": metadata or {},
    }
