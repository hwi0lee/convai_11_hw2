"""Deterministic constraint scheduler.

The LLM never places blocks itself. It only translates language into state changes;
this module guarantees the hard constraints:
  * no overlap with calendar events / user-blocked intervals / other blocks
  * blocks only inside availability windows, never in protected sleep hours, never in the past
  * every block of a task ends before that task's deadline
  * per-day work cap
Placement is Earliest-Deadline-First (EDF) over 30-minute slots, which is optimal for
feasibility in the single-resource preemptive case. On re-planning we first try to keep
still-valid blocks from the previous plan (stability), and only fall back to a full
re-plan if keeping them would cause a deadline miss.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Set, Tuple

from .models import (SLEEP_END, SLEEP_START, SLOT_MIN, Block, State, Task,
                     fmt_dt, overlaps)

SLOT = timedelta(minutes=SLOT_MIN)


def _ceil_slot(d: datetime) -> datetime:
    d = d.replace(second=0, microsecond=0)
    rem = d.minute % SLOT_MIN
    return d if rem == 0 else d + timedelta(minutes=SLOT_MIN - rem)


def _hm(day: date, s: str) -> datetime:
    if s == "24:00":
        return datetime.combine(day, time(0, 0)) + timedelta(days=1)
    h, m = map(int, s.split(":"))
    return datetime.combine(day, time(h, m))


def _in_sleep(slot_start: datetime) -> bool:
    t = slot_start.time()
    return SLEEP_START <= t < SLEEP_END


def window_slots(state: State) -> List[datetime]:
    """All slot starts inside availability windows (ignoring busy intervals)."""
    start = _ceil_slot(state.now)
    horizon_end = datetime.combine(state.now.date(), time(0)) + timedelta(days=state.prefs.horizon_days + 1)
    slots: Set[datetime] = set()
    for day in state.day_list():
        for ws, we in state.prefs.windows:
            s, e = _hm(day, ws), _hm(day, we)
            cur = s
            while cur + SLOT <= e:
                slots.add(cur)
                cur += SLOT
    for x in state.extra_windows:
        cur = _ceil_slot(x.start)
        while cur + SLOT <= x.end:
            slots.add(cur)
            cur += SLOT
    return sorted(s for s in slots if s >= start and s < horizon_end and not _in_sleep(s))


def free_slots(state: State) -> List[datetime]:
    busy = [(e.start, e.end) for e in state.events] + [(b.start, b.end) for b in state.blocked]
    out = []
    for s in window_slots(state):
        if not any(overlaps(s, s + SLOT, bs, be) for bs, be in busy):
            out.append(s)
    return out


@dataclass
class Dropped:
    task_id: str
    start: datetime
    end: datetime
    reason: str


@dataclass
class ScheduleResult:
    plan: List[Block]
    kept_min: int = 0
    dropped: List[Dropped] = field(default_factory=list)
    shortfall: Dict[str, int] = field(default_factory=dict)  # task_id -> unscheduled minutes
    full_replan: bool = False


def _drop_reason(state: State, b: Block, task: Optional[Task], free: Set[datetime]) -> Optional[str]:
    if task is None:
        return "작업이 삭제됨"
    if b.start < state.now:
        return "이미 지난 시간"
    if b.end > task.deadline:
        return f"마감({fmt_dt(task.deadline)}) 이후 블록"
    cur = b.start
    while cur < b.end:
        if cur not in free:
            for bl in state.blocked:
                if overlaps(cur, cur + SLOT, bl.start, bl.end):
                    return f"사용자가 막은 시간과 충돌({bl.reason or fmt_dt(bl.start)})"
            for ev in state.events:
                if overlaps(cur, cur + SLOT, ev.start, ev.end):
                    return "캘린더 일정과 충돌"
            return "작업 가능 시간대 밖"
        cur += SLOT
    return None


def _work_by_day(blocks: List[Block]) -> Dict[date, int]:
    d: Dict[date, int] = defaultdict(int)
    for b in blocks:
        if b.status in ("planned", "done"):
            d[b.start.date()] += b.minutes
    return d


def _merge(slots_by_task: Dict[str, List[datetime]]) -> List[Block]:
    blocks = []
    for tid, slots in slots_by_task.items():
        slots = sorted(slots)
        if not slots:
            continue
        s = prev = slots[0]
        for x in slots[1:]:
            if x == prev + SLOT:
                prev = x
                continue
            blocks.append(Block(tid, s, prev + SLOT))
            s = prev = x
        blocks.append(Block(tid, s, prev + SLOT))
    return blocks


def _split_long(blocks: List[Block], state: State) -> List[Block]:
    out = []
    for b in blocks:
        t = state.task_by_id(b.task_id)
        mx = timedelta(minutes=t.max_block_min if t else 120)
        cur = b.start
        while cur < b.end:
            nxt = min(cur + mx, b.end)
            out.append(Block(b.task_id, cur, nxt))
            cur = nxt
    return out


def _schedule_once(state: State, keep_previous: bool) -> ScheduleResult:
    """Stability-aware EDF.

    Tasks are processed in deadline order. Each task first re-uses its own still-valid
    slots from the previous plan, then fills free slots that nobody owned before, and only
    if that is not enough does it take slots previously owned by later-deadline tasks
    (those tasks are re-placed afterwards). Without previous blocks this is plain EDF.
    """
    history = [b for b in state.plan if b.status != "planned"]
    prev = sorted([b for b in state.plan if b.status == "planned"], key=lambda b: b.start)
    free = set(free_slots(state))
    cap = state.prefs.max_work_min_per_day
    day_used = _work_by_day(history)

    assigned: Dict[str, List[datetime]] = defaultdict(list)
    owner: Dict[datetime, str] = {}
    need = {t.id: t.remaining_min for t in state.tasks}
    res = ScheduleResult(plan=[])

    # previous ownership of still-valid slots; invalid previous blocks get a reason
    prev_owner: Dict[datetime, str] = {}
    reason_of: Dict[Tuple[str, datetime], str] = {}
    if keep_previous:
        for b in prev:
            task = state.task_by_id(b.task_id)
            reason = _drop_reason(state, b, task, free)
            cur = b.start
            while cur < b.end:
                if reason:
                    reason_of[(b.task_id, cur)] = reason
                else:
                    prev_owner[cur] = b.task_id
                cur += SLOT

    def take(tid: str, s: datetime):
        free.discard(s)
        owner[s] = tid
        assigned[tid].append(s)
        need[tid] -= SLOT_MIN
        day_used[s.date()] += SLOT_MIN

    def run_len(tid: str, s: datetime) -> int:
        run, p = 0, s - SLOT
        while owner.get(p) == tid:
            run += 1
            p -= SLOT
        n = s + SLOT
        while owner.get(n) == tid:
            run += 1
            n += SLOT
        return run

    # daily capacity "reserved" by previous blocks of tasks not processed yet
    reserved: Dict[date, int] = defaultdict(int)
    for s_, tid_ in prev_owner.items():
        reserved[s_.date()] += SLOT_MIN

    order = sorted(state.tasks, key=lambda t: (t.deadline, t.priority, t.name))
    for t in order:
        max_run = max(1, t.max_block_min // SLOT_MIN)
        for s_ in (k for k, v in prev_owner.items() if v == t.id):
            reserved[s_.date()] -= SLOT_MIN
        # 1) keep own previous slots
        for s in sorted(k for k, v in prev_owner.items() if v == t.id):
            if need[t.id] <= 0:
                reason_of[(t.id, s)] = "필요 시간이 줄어 불필요해짐"
            elif s in free and day_used[s.date()] + SLOT_MIN <= cap:
                take(t.id, s)
                res.kept_min += SLOT_MIN
            elif s in free:
                reason_of[(t.id, s)] = "하루 최대 작업시간 초과"
        # 2) unowned free slots, 3) slots of later-deadline tasks if still short
        for allow_steal in (False, True):
            for s in sorted(free):
                if need[t.id] <= 0:
                    break
                if s not in free or s + SLOT > t.deadline:
                    continue
                if not allow_steal and s in prev_owner and prev_owner[s] != t.id:
                    continue
                # first pass also respects capacity reserved by other tasks' kept blocks
                used = day_used[s.date()] + (0 if allow_steal else reserved[s.date()])
                if used + SLOT_MIN > cap or run_len(t.id, s) >= max_run:
                    continue
                take(t.id, s)
        if need[t.id] > 0:
            res.shortfall[t.id] = need[t.id]

    # explain every previous slot that is not in the new plan
    for s, tid in prev_owner.items():
        if owner.get(s) != tid and (tid, s) not in reason_of:
            other = state.task_by_id(owner[s]) if s in owner else None
            reason_of[(tid, s)] = f"마감이 더 급한 '{other.name}'에 시간 양보" if other else "재배치"
    by_reason: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)
    for (tid, s), r in reason_of.items():
        by_reason[(tid, r)].append(s)
    for (tid, r), slots in by_reason.items():
        for b in _merge({tid: slots}):
            res.dropped.append(Dropped(tid, b.start, b.end, r))
    res.dropped.sort(key=lambda d: d.start)

    res.plan = history + _split_long(_merge(assigned), state)
    res.plan.sort(key=lambda b: b.start)
    return res


def schedule(state: State, keep_previous: bool = True) -> ScheduleResult:
    res = _schedule_once(state, keep_previous)
    if keep_previous and res.shortfall:
        # safety net: never let stability cost a deadline
        fresh = _schedule_once(state, keep_previous=False)
        if sum(fresh.shortfall.values()) < sum(res.shortfall.values()):
            fresh.full_replan = True
            fresh.dropped = res.dropped
            return fresh
    return res


# ---------------------------------------------------------------------------
# Diff between two plans (minute-level, per task) -> used for explanations/stability
# ---------------------------------------------------------------------------

def _slotset(blocks: List[Block], now: datetime) -> Set[Tuple[str, datetime]]:
    out = set()
    for b in blocks:
        if b.status != "planned":
            continue
        cur = max(b.start, _ceil_slot(now)) if b.start < now else b.start
        while cur < b.end:
            out.add((b.task_id, cur))
            cur += SLOT
    return out


def _to_intervals(pairs: Set[Tuple[str, datetime]]) -> List[Tuple[str, datetime, datetime]]:
    by: Dict[str, List[datetime]] = defaultdict(list)
    for tid, s in pairs:
        by[tid].append(s)
    return [(b.task_id, b.start, b.end) for b in _merge(by)]


def plan_diff(old: List[Block], new: List[Block], now: datetime) -> Dict[str, List[Tuple[str, datetime, datetime]]]:
    o, n = _slotset(old, now), _slotset(new, now)
    return {
        "removed": sorted(_to_intervals(o - n), key=lambda x: x[1]),
        "added": sorted(_to_intervals(n - o), key=lambda x: x[1]),
        "unchanged_min": len(o & n) * SLOT_MIN,
        "old_future_min": len(o) * SLOT_MIN,
    }
