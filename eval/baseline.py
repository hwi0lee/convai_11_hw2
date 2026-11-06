"""Baseline: plain ChatGPT given exactly the same information as text, asked for a JSON plan.

Two variants:
  baseline      - direct answer
  baseline_cot  - same, but asked to reason step by step (in a 'reasoning' field) first
"""
from __future__ import annotations

import json
from typing import Dict, List

from planner.models import State, fmt_dt, weekday_ko

RULES = """규칙:
- 작업 블록은 30분 단위, 'YYYY-MM-DD HH:MM' 형식.
- 매일 작업 가능 시간대 안에서만 배치. 현재 시각 이전에는 배치 불가. 00:00~07:00은 수면 시간이라 불가.
- 캘린더 일정 및 사용자가 안 된다고 한 시간과 겹치면 안 됨. 블록끼리도 겹치면 안 됨.
- 각 작업의 모든 블록은 마감 전에 끝나야 함. 각 작업의 남은 시간만큼 정확히 배치.
- 하루 총 작업시간은 하루 최대 작업시간을 넘으면 안 됨. 한 블록은 최대 2시간.
- 시간이 부족해 모두 배치할 수 없으면 infeasible=true로 하고, 배치 못 한 시간을 unscheduled에 적기."""

FORMAT = """반드시 아래 JSON 형식으로만 답하세요:
{%s"explanation": "사용자에게 보여줄 한국어 설명(무엇을 왜 바꿨는지)",
 "infeasible": true/false,
 "unscheduled": [{"task": "작업 이름", "hours": 숫자}],
 "blocks": [{"task": "작업 이름", "start": "YYYY-MM-DD HH:MM", "end": "YYYY-MM-DD HH:MM"}]}
blocks에는 현재 시각 이후의 전체 계획을 모두 포함하세요."""


def describe(s: State) -> str:
    ev = "\n".join(f"- {e.start:%Y-%m-%d}({weekday_ko(e.start.date())}) {e.start:%H:%M}-{e.end:%H:%M} 일정"
                   for e in sorted(s.events, key=lambda e: e.start))
    bl = "".join(f"\n- {b.start:%Y-%m-%d %H:%M} ~ {b.end:%Y-%m-%d %H:%M} 작업 불가({b.reason})" for b in s.blocked if b.end > s.now)
    tk = "\n".join(f"- {t.name}: 남은 {t.remaining_min/60:g}시간, 마감 {fmt_dt(t.deadline)}({weekday_ko(t.deadline.date())}), 우선순위 {t.priority}"
                   for t in s.tasks)
    return f"""현재 시각: {fmt_dt(s.now)} ({weekday_ko(s.now.date())}요일)
매일 작업 가능 시간대: {', '.join(f'{a}-{b}' for a, b in s.prefs.windows)}
하루 최대 작업시간: {s.prefs.max_work_min_per_day/60:g}시간

캘린더 일정:
{ev}{bl}

작업 목록:
{tk}

위 정보로 이번 주 작업 계획을 세워 주세요."""


class Baseline:
    def __init__(self, client, model: str, cot: bool = False):
        self.client, self.model, self.cot = client, model, cot
        sys = "당신은 사용자의 일정을 관리해 주는 어시스턴트입니다.\n" + RULES + "\n\n"
        sys += FORMAT % ('"reasoning": "단계별로 가용 시간 계산, 충돌·마감 확인 과정",\n ' if cot else "")
        if cot:
            sys += "\n답하기 전에 reasoning 필드에서 단계적으로 생각하세요: 날짜별 가용 시간을 계산하고, 마감 순서대로 배치하고, 충돌과 마감을 다시 검토하세요."
        self.messages: List[Dict] = [{"role": "system", "content": sys}]

    def _ask(self, content: str) -> Dict:
        self.messages.append({"role": "user", "content": content})
        r = self.client.chat.completions.create(model=self.model, messages=self.messages, temperature=0,
                                                response_format={"type": "json_object"})
        txt = r.choices[0].message.content or "{}"
        self.messages.append({"role": "assistant", "content": txt})
        try:
            out = json.loads(txt)
        except json.JSONDecodeError:
            out = {"blocks": [], "explanation": txt, "infeasible": False, "_parse_error": True}
        out.setdefault("blocks", [])
        return out

    def start(self, s: State) -> Dict:
        return self._ask(describe(s))

    def follow_up(self, now_str: str, weekday: str, utterance: str) -> Dict:
        return self._ask(f"(현재 시각: {now_str} {weekday}요일)\n{utterance}\n\n변경 사항을 반영해 현재 시각 이후의 전체 계획을 같은 JSON 형식으로 다시 출력하세요.")
