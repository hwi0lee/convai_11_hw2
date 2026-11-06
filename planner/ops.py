"""State-changing operations the LLM may call (function calling).

Every op validates its arguments and enforces safety rules (sleep protection,
daily work cap, sane durations) before touching the state. Ops return a short
message that is fed back to the LLM as the tool result.
"""
from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (HARD_MAX_WORK_MIN_PER_DAY, SLEEP_END, Blocked, State, Task,
                     fmt_dt, overlaps, parse_dt)


class OpError(Exception):
    pass


def _task(state: State, name: str) -> Task:
    t = state.task_by_name(name)
    if not t:
        names = ", ".join(x.name for x in state.tasks) or "(없음)"
        raise OpError(f"'{name}' 작업을 찾을 수 없습니다. 현재 작업: {names}")
    return t


def _hours(x: Any, lo: float = 0.0, hi: float = 100.0, what: str = "시간") -> float:
    try:
        h = float(x)
    except (TypeError, ValueError):
        raise OpError(f"{what} 값이 숫자가 아닙니다: {x!r}")
    if not (lo <= h <= hi):
        raise OpError(f"{what} 값 {h}이(가) 허용 범위({lo}~{hi})를 벗어납니다.")
    return h


def _round30(minutes: float) -> int:
    return int(round(minutes / 30.0)) * 30


def _fit_horizon(state: State) -> None:
    if state.tasks:
        last = max(t.deadline for t in state.tasks)
        days = (last.date() - state.now.date()).days
        state.prefs.horizon_days = min(21, max(7, days))


# ---------------------------------------------------------------------------

def add_task(state: State, name: str, duration_hours: float, deadline: str,
             priority: int = 2, max_block_hours: float = 2.0) -> str:
    if not name.strip():
        raise OpError("작업 이름이 비어 있습니다.")
    if state.task_by_name(name) and state.task_by_name(name).name == name.strip():
        raise OpError(f"'{name}' 작업이 이미 있습니다. 수정하려면 update_task를 사용하세요.")
    dur = _round30(_hours(duration_hours, 0.5, 100, "예상 소요시간") * 60)
    dl = parse_dt(deadline)
    if len(str(deadline).strip()) == 10:   # date only -> end of day
        dl = dl.replace(hour=23, minute=59)
    if dl <= state.now:
        raise OpError(f"마감 {fmt_dt(dl)}이(가) 현재 시각 {fmt_dt(state.now)}보다 이전입니다.")
    pr = int(priority) if str(priority).isdigit() and int(priority) in (1, 2, 3) else 2
    mb = _round30(_hours(max_block_hours, 0.5, 4, "최대 연속 작업시간") * 60)
    state.tasks.append(Task(name.strip(), dur, dl, pr, mb))
    _fit_horizon(state)
    return f"작업 추가: {name} ({dur/60:g}시간, 마감 {fmt_dt(dl)}, 우선순위 {pr})"


def update_task(state: State, name: str, duration_hours: Optional[float] = None,
                add_hours: Optional[float] = None, deadline: Optional[str] = None,
                priority: Optional[int] = None) -> str:
    t = _task(state, name)
    msgs = []
    if duration_hours is not None and float(duration_hours) <= 0:   # "0시간" == drop the task
        return remove_task(state, name)
    if duration_hours is not None:
        t.duration_min = _round30(_hours(duration_hours, 0.5, 100, "예상 소요시간") * 60) + 0
        t.duration_min = max(t.duration_min, 0)
        msgs.append(f"총 소요시간 {t.duration_min/60:g}시간")
    if add_hours is not None:
        delta = _round30(_hours(add_hours, -100, 100, "추가 시간") * 60)
        t.duration_min = max(t.done_min, t.duration_min + delta)
        msgs.append(f"소요시간 {delta/60:+g}시간 → 총 {t.duration_min/60:g}시간")
    if deadline is not None:
        dl = parse_dt(deadline)
        if len(str(deadline).strip()) == 10:
            dl = dl.replace(hour=23, minute=59)
        if dl <= state.now:
            raise OpError(f"새 마감 {fmt_dt(dl)}이(가) 현재 시각보다 이전입니다.")
        t.deadline = dl
        msgs.append(f"마감 {fmt_dt(dl)}")
    if priority is not None:
        if int(priority) not in (1, 2, 3):
            raise OpError("우선순위는 1(높음)~3(낮음)입니다.")
        t.priority = int(priority)
        msgs.append(f"우선순위 {t.priority}")
    _fit_horizon(state)
    return f"작업 수정: {t.name} ({', '.join(msgs) or '변경 없음'}), 남은 시간 {t.remaining_min/60:g}시간"


def remove_task(state: State, name: str) -> str:
    t = _task(state, name)
    state.tasks = [x for x in state.tasks if x.id != t.id]
    state.plan = [b for b in state.plan if not (b.task_id == t.id and b.status == "planned")]
    return f"작업 삭제: {t.name}"


def complete_task(state: State, name: str) -> str:
    t = _task(state, name)
    t.done_min = t.duration_min
    return f"작업 완료 처리: {t.name}"


def block_time(state: State, start: str, end: str, reason: str = "") -> str:
    s, e = parse_dt(start), parse_dt(end)
    if e <= s:
        raise OpError(f"종료 시각({fmt_dt(e)})이 시작 시각({fmt_dt(s)})보다 빠릅니다.")
    if e <= state.now:
        raise OpError("이미 지난 시간은 막을 필요가 없습니다.")
    state.blocked.append(Blocked(s, e, reason))
    return f"시간 막음: {fmt_dt(s)} ~ {fmt_dt(e)} ({reason or '사유 없음'})"


def unblock_time(state: State, start: str, end: str) -> str:
    s, e = parse_dt(start), parse_dt(end)
    before = len(state.blocked)
    state.blocked = [b for b in state.blocked if not overlaps(b.start, b.end, s, e)]
    return f"막아둔 시간 {before - len(state.blocked)}건 해제"


def add_availability(state: State, start: str, end: str) -> str:
    s, e = parse_dt(start), parse_dt(end)
    if e <= s:
        raise OpError("종료 시각이 시작 시각보다 빠릅니다.")
    warn = ""
    # Safety: never schedule into protected sleep hours (00:00-07:00)
    if s.time() < SLEEP_END:
        s = datetime.combine(s.date(), SLEEP_END)
        warn = " (수면 보호 시간 00:00~07:00은 제외했습니다)"
    midnight = datetime.combine(s.date() + timedelta(days=1), time(0, 0))
    if e > midnight:
        e = midnight
        warn = " (자정 이후 수면 보호 시간은 제외했습니다)"
    if e <= s:
        raise OpError("요청한 시간은 모두 수면 보호 시간(00:00~07:00)이라 추가하지 않았습니다. 충분한 수면을 권장합니다.")
    state.extra_windows.append(Blocked(s, e, "추가 가능 시간"))
    return f"작업 가능 시간 추가: {fmt_dt(s)} ~ {fmt_dt(e)}{warn}"


def report_missed(state: State, date: str, task_name: Optional[str] = None) -> str:
    d = parse_dt(date[:10]).date()
    tid = _task(state, task_name).id if task_name else None
    n, mins = 0, 0
    for b in state.plan:
        if b.status == "done" and b.start.date() == d and (tid is None or b.task_id == tid):
            b.status = "missed"
            t = state.task_by_id(b.task_id)
            if t:
                t.done_min = max(0, t.done_min - b.minutes)
            n += 1
            mins += b.minutes
    if n == 0:
        return f"{d}에 완료 처리된 블록이 없습니다(이미 반영되었거나 계획이 없었음)."
    return f"{d} 미완료 반영: 블록 {n}개, {mins/60:g}시간을 다시 배치 대상으로 되돌림"


def report_progress(state: State, task_name: str, done_hours: float, date: Optional[str] = None) -> str:
    t = _task(state, task_name)
    h = _hours(done_hours, 0, 24, "완료한 시간")
    if date:
        d = parse_dt(date[:10]).date()
        for b in state.plan:
            if b.status == "done" and b.task_id == t.id and b.start.date() == d:
                b.status = "missed"
                t.done_min = max(0, t.done_min - b.minutes)
    t.done_min = min(t.duration_min, t.done_min + _round30(h * 60))
    return f"진행 반영: {t.name} {h:g}시간 완료, 남은 시간 {t.remaining_min/60:g}시간"


_HM = re.compile(r"^\d{2}:\d{2}$")


def set_preferences(state: State, max_hours_per_day: Optional[float] = None,
                    windows: Optional[List[List[str]]] = None) -> str:
    msgs = []
    if max_hours_per_day is not None:
        h = _hours(max_hours_per_day, 0.5, 24, "하루 최대 작업시간")
        if h * 60 > HARD_MAX_WORK_MIN_PER_DAY:
            msgs.append(f"요청한 하루 {h:g}시간은 건강을 위해 상한 {HARD_MAX_WORK_MIN_PER_DAY/60:g}시간으로 제한했습니다")
            h = HARD_MAX_WORK_MIN_PER_DAY / 60
        state.prefs.max_work_min_per_day = _round30(h * 60)
        msgs.append(f"하루 최대 {h:g}시간")
    if windows is not None:
        clean = []
        for w in windows:
            if len(w) != 2 or not all(_HM.match(x) for x in w):
                raise OpError(f"시간대 형식 오류: {w} (예: ['09:00','12:00'])")
            a, b = w
            if a < SLEEP_END.strftime("%H:%M"):
                a = SLEEP_END.strftime("%H:%M")
                msgs.append("수면 보호 시간(00~07시)은 제외")
            if b <= a and b != "24:00":
                continue
            clean.append([a, b])
        if not clean:
            raise OpError("유효한 작업 가능 시간대가 없습니다.")
        state.prefs.windows = clean
        msgs.append("작업 시간대 " + ", ".join(f"{a}-{b}" for a, b in clean))
    return "설정 변경: " + "; ".join(msgs)


OPS: Dict[str, Callable[..., str]] = {
    "add_task": add_task,
    "update_task": update_task,
    "remove_task": remove_task,
    "complete_task": complete_task,
    "block_time": block_time,
    "unblock_time": unblock_time,
    "add_availability": add_availability,
    "report_missed": report_missed,
    "report_progress": report_progress,
    "set_preferences": set_preferences,
}


def apply_op(state: State, name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
    fn = OPS.get(name)
    if fn is None:
        return False, f"알 수 없는 연산: {name}"
    args = {k: v for k, v in args.items() if v is not None}
    try:
        return True, fn(state, **args)
    except OpError as e:
        return False, str(e)
    except (ValueError, TypeError) as e:
        return False, f"인자 오류: {e}"
