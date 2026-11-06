"""System-agnostic plan checker. A plan = list of {"task", "start", "end"} dicts.

Every block is validated against the *gold* state (not the system's own belief), so a
system that misunderstood the user ("내일" -> wrong date) is penalised automatically.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from planner.models import SLOT_MIN, State, overlaps, parse_dt
from planner.solver import _ceil_slot, window_slots

SLOT = timedelta(minutes=SLOT_MIN)


def _norm(s: str) -> str:
    return "".join(str(s).lower().split())


def match_task(state: State, name: str):
    n = _norm(name)
    for t in state.tasks:
        if _norm(t.name) == n:
            return t
    cands = [t for t in state.tasks if n and (n in _norm(t.name) or _norm(t.name) in n)]
    return cands[0] if len(cands) == 1 else None


def parse_plan(raw: List[Dict]) -> Tuple[List[Tuple[str, datetime, datetime]], int]:
    out, bad = [], 0
    for b in raw or []:
        try:
            s, e = parse_dt(b["start"]), parse_dt(b["end"])
            out.append((str(b.get("task", "")), s, e))
        except Exception:  # noqa: BLE001
            bad += 1
    return out, bad


def check(gold: State, raw_plan: List[Dict]) -> Dict:
    blocks, malformed = parse_plan(raw_plan)
    win = set(window_slots(gold))           # also excludes past & sleep hours
    busy = [(e.start, e.end) for e in gold.events] + [(b.start, b.end) for b in gold.blocked]
    v = defaultdict(int)                    # violation counters (per block)
    valid_min = defaultdict(int)
    used_slots: Dict[datetime, int] = defaultdict(int)
    day_min = defaultdict(int)
    future_blocks = 0

    for name, s, e in blocks:
        if e <= gold.now:                   # past blocks are ignored (history)
            continue
        future_blocks += 1
        t = match_task(gold, name)
        ok = True
        if e <= s:
            v["malformed"] += 1
            continue
        if t is None:
            v["unknown_task"] += 1
            ok = False
        slots = []
        cur = s
        while cur < e:
            slots.append(cur)
            cur += SLOT
        if s < gold.now:
            v["in_past"] += 1
            ok = False
        if any(sl not in win for sl in slots if sl >= gold.now):
            v["outside_window"] += 1
            ok = False
        if any(overlaps(s, e, bs, be) for bs, be in busy):
            v["conflict"] += 1
            ok = False
        if t is not None and e > t.deadline:
            v["after_deadline"] += 1
            ok = False
        if any(used_slots[sl] for sl in slots):
            v["self_overlap"] += 1
            ok = False
        for sl in slots:
            used_slots[sl] += 1
            day_min[sl.date()] += SLOT_MIN
        if ok and t is not None:
            valid_min[t.id] += int((e - s).total_seconds() // 60)

    cap_days = sum(1 for d, m in day_min.items() if m > gold.prefs.max_work_min_per_day)
    req = {t.id: t.remaining_min for t in gold.tasks if t.remaining_min > 0}
    covered = {tid: min(valid_min.get(tid, 0), r) for tid, r in req.items()}
    over = sum(max(0, valid_min.get(tid, 0) - r) for tid, r in req.items())
    n_viol_blocks = sum(v[k] for k in ("unknown_task", "in_past", "outside_window", "conflict", "after_deadline", "self_overlap"))
    hard_ok = n_viol_blocks == 0 and cap_days == 0 and malformed == 0 and v["malformed"] == 0
    total_req = sum(req.values())
    return {
        "n_blocks": future_blocks,
        "violations": dict(v),
        "cap_exceeded_days": cap_days,
        "malformed": malformed + v["malformed"],
        "hard_ok": hard_ok,
        "conflict_rate": v["conflict"] / future_blocks if future_blocks else 0.0,
        "deadline_violation_rate": v["after_deadline"] / future_blocks if future_blocks else 0.0,
        "coverage": (sum(covered.values()) / total_req) if total_req else 1.0,
        "tasks_satisfied": sum(1 for tid, r in req.items() if covered[tid] >= r) / len(req) if req else 1.0,
        "over_alloc_min": over,
        "complete_valid": hard_ok and all(covered[tid] >= r for tid, r in req.items()),
    }


def _valid_slots(gold: State, raw_plan: List[Dict]) -> Set[Tuple[str, datetime]]:
    blocks, _ = parse_plan(raw_plan)
    win = set(window_slots(gold))
    busy = [(e.start, e.end) for e in gold.events] + [(b.start, b.end) for b in gold.blocked]
    out = set()
    for name, s, e in blocks:
        t = match_task(gold, name)
        if t is None:
            continue
        cur = max(s, _ceil_slot(gold.now))
        while cur < e:
            if cur in win and cur + SLOT <= t.deadline and not any(overlaps(cur, cur + SLOT, bs, be) for bs, be in busy):
                out.add((t.id, cur))
            cur += SLOT
    return out


def stability(gold_new: State, prev_plan: List[Dict], new_plan: List[Dict]) -> Optional[float]:
    """Share of the previous plan's blocks that are still valid under the new gold state
    and were kept unchanged (same task, same time). Higher = less unnecessary churn."""
    prev = _valid_slots(gold_new, prev_plan)
    if not prev:
        return None
    blocks, _ = parse_plan(new_plan)
    new = set()
    for name, s, e in blocks:
        t = match_task(gold_new, name)
        if t is None:
            continue
        cur = s
        while cur < e:
            new.add((t.id, cur))
            cur += SLOT
    return len(prev & new) / len(prev)


def state_match(gold: State, mine: State) -> Dict:
    """Did the system's internal state end up equal to the gold state? (op-parsing accuracy)"""
    errs = []
    for g in gold.tasks:
        m = match_task(mine, g.name)
        if m is None:
            errs.append(f"missing_task:{g.name}")
            continue
        if m.remaining_min != g.remaining_min:
            errs.append(f"remaining:{g.name}:{m.remaining_min}!={g.remaining_min}")
        if abs((m.deadline - g.deadline).total_seconds()) > 60:
            errs.append(f"deadline:{g.name}:{m.deadline:%m-%d %H:%M}!={g.deadline:%m-%d %H:%M}")
    for m in mine.tasks:
        if m.remaining_min > 0 and match_task(gold, m.name) is None:
            errs.append(f"extra_task:{m.name}")
    # blocked time: compare the set of window slots that are unavailable
    win = window_slots(gold)
    gb = {s for s in win if any(overlaps(s, s + SLOT, b.start, b.end) for b in gold.blocked)}
    mb = {s for s in win if any(overlaps(s, s + SLOT, b.start, b.end) for b in mine.blocked)}
    if gb != mb:
        errs.append(f"blocked:missing={len(gb - mb)},extra={len(mb - gb)}")
    return {"ok": not errs, "errors": errs}


def block_violations(gold: State, raw_plan: List[Dict]) -> Tuple[List[Dict], Set]:
    """Per-block version of `check` (same rules), for highlighting in the demo UI.
    Returns ([{task, start, end, reasons}], set of dates over the daily cap)."""
    blocks, _ = parse_plan(raw_plan)
    win = set(window_slots(gold))
    busy = [(e.start, e.end) for e in gold.events] + [(b.start, b.end) for b in gold.blocked]
    used: Dict[datetime, int] = defaultdict(int)
    day_min: Dict = defaultdict(int)
    out = []
    for name, s, e in sorted(blocks, key=lambda x: x[1]):
        if e <= gold.now or e <= s:
            continue
        t = match_task(gold, name)
        slots = []
        cur = s
        while cur < e:
            slots.append(cur)
            cur += SLOT
        reasons = []
        if t is None:
            reasons.append("없는 작업")
        if s < gold.now:
            reasons.append("과거 시간")
        if any(overlaps(s, e, bs, be) for bs, be in busy):
            reasons.append("일정/막은 시간과 충돌")
        if any(sl not in win for sl in slots if sl >= gold.now):
            reasons.append("작업 가능 시간대 밖")
        if t is not None and e > t.deadline:
            reasons.append("마감 이후")
        if any(used[sl] for sl in slots):
            reasons.append("블록끼리 겹침")
        for sl in slots:
            used[sl] += 1
            day_min[sl.date()] += SLOT_MIN
        out.append({"task": name, "start": s, "end": e, "reasons": reasons})
    over = {d for d, m in day_min.items() if m > gold.prefs.max_work_min_per_day}
    return out, over
