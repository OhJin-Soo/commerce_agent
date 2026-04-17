"""실시간 환율 조회 유틸리티."""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_FALLBACK_RATE = 16.0  # 1 INR ≈ 16 KRW (API 실패 시 사용)


async def fetch_inr_to_krw() -> float:
    """open.er-api.com 에서 INR→KRW 환율을 가져온다.

    실패 시 fallback 값을 반환하며 서버 시작을 중단시키지 않는다.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://open.er-api.com/v6/latest/INR")
            resp.raise_for_status()
            rate = float(resp.json()["rates"]["KRW"])
        logger.info("Exchange rate  INR→KRW: %.4f", rate)
        return rate
    except Exception as exc:
        logger.warning(
            "환율 조회 실패 (%s) — fallback %.1f 사용", exc, _FALLBACK_RATE
        )
        return _FALLBACK_RATE
