"""에이전트 공용 유틸리티."""
from __future__ import annotations

import re


def strip_thinking(text: str) -> str:
    """DeepSeek-R1 등 추론 모델이 출력하는 <think>...</think> 블록을 제거한다.

    - 일반 모델(llama3.1:8b 등)은 이 태그를 출력하지 않으므로 no-op 이다.
    - 중첩 태그나 대소문자 변형은 없다고 가정한다 (DeepSeek 스펙상 항상 소문자).
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
