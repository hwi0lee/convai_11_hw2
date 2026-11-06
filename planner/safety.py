"""Safety layer: moderation, crisis response, overwork guard, privacy-preserving context."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

CRISIS_MSG = (
    "많이 힘드신 것 같아 걱정돼요. 일정보다 지금 당신의 안전이 더 중요합니다.\n"
    "혼자 감당하지 마시고 지금 바로 도움을 받아보세요.\n"
    "- 자살예방상담전화 **109** (24시간)\n"
    "- 정신건강위기상담 **1577-0199**\n"
    "- 긴급한 위험이 있다면 **112 / 119**\n"
    "원하시면 이번 주 일정을 최대한 가볍게 줄이는 것도 함께 도와드릴게요."
)

REFUSAL_MSG = "죄송하지만 그 요청은 도와드릴 수 없어요. 일정·작업 계획과 관련된 이야기라면 언제든 도와드릴게요."

_OVERWORK = re.compile(r"(밤\s*새|밤샘|철야|잠\s*(안|못)\s*자|안\s*자고|새벽\s*[1-6]\s*시까지|24\s*시간\s*(내내|동안)|쉬지\s*않고)")


@dataclass
class ModResult:
    flagged: bool
    self_harm: bool = False
    categories: Optional[List[str]] = None
    error: Optional[str] = None


def moderate(client, text: str) -> ModResult:
    """OpenAI moderation. Fails open (logs error) so an outage does not block planning,
    but the rule-based guards below still apply."""
    try:
        r = client.moderations.create(model="omni-moderation-latest", input=text).results[0]
        cats = [k for k, v in r.categories.model_dump().items() if v]
        sh = any(c.startswith("self_harm") or c.startswith("self-harm") for c in cats)
        return ModResult(r.flagged, sh, cats)
    except Exception as e:  # noqa: BLE001
        return ModResult(False, error=str(e))


def overwork_request(text: str) -> bool:
    return bool(_OVERWORK.search(text))


OVERWORK_NOTE = (
    "[안전 규칙] 사용자가 수면을 줄이거나 과도한 작업을 원합니다. 수면 보호 시간(00~07시)과 "
    "하루 최대 작업시간 상한은 변경할 수 없음을 부드럽게 알리고, 대신 범위 축소·마감 조정 같은 현실적 대안을 제시하세요."
)


def sanitize_title(title: str, max_len: int = 30) -> str:
    """Event titles are untrusted (could contain prompt-injection). Strip control chars, truncate."""
    t = re.sub(r"[\r\n\t`<>{}\[\]]", " ", title or "")
    return t[:max_len].strip() or "일정"
