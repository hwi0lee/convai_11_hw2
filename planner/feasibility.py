"""Workload-vs-capacity analysis and repair suggestions for infeasible plans."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List

from .models import SLOT_MIN, State, fmt_dt
from .solver import ScheduleResult, _work_by_day, free_slots, schedule


@dataclass
class DeadlineRow:
    deadline: str
    tasks: List[str]
    demand_min: int      # cumulative remaining effort due by this deadline
    capacity_min: int    # usable free time until this deadline (respecting daily cap)
    shortfall_min: int


@dataclass
class Report:
    feasible: bool
    rows: List[DeadlineRow]
    total_demand_min: int
    total_capacity_min: int
    utilization: float
    shortfall: Dict[str, int]                  # task name -> unscheduled minutes
    suggestions: List[str] = field(default_factory=list)


def capacity_until(state: State, until) -> int:
    cap = state.prefs.max_work_min_per_day
    used = _work_by_day([b for b in state.plan if b.status != "planned"])
    per_day: Dict = defaultdict(int)
    for s in free_slots(state):
        if s + timedelta(minutes=SLOT_MIN) <= until:
            per_day[s.date()] += SLOT_MIN
    return sum(max(0, min(v, cap - used.get(d, 0))) for d, v in per_day.items())


def analyze(state: State, res: ScheduleResult) -> Report:
    rows: List[DeadlineRow] = []
    tasks = sorted([t for t in state.tasks if t.remaining_min > 0], key=lambda t: t.deadline)
    demand = 0
    i = 0
    while i < len(tasks):
        dl = tasks[i].deadline
        group = []
        while i < len(tasks) and tasks[i].deadline == dl:
            group.append(tasks[i])
            demand += tasks[i].remaining_min
            i += 1
        cap = capacity_until(state, dl)
        rows.append(DeadlineRow(fmt_dt(dl), [t.name for t in group], demand, cap, max(0, demand - cap)))

    horizon_end = state.now + timedelta(days=state.prefs.horizon_days + 1)
    total_cap = capacity_until(state, horizon_end)
    total_dem = sum(t.remaining_min for t in tasks)
    short = {state.task_by_id(k).name: v for k, v in res.shortfall.items() if state.task_by_id(k)}
    rep = Report(
        feasible=not res.shortfall,
        rows=rows,
        total_demand_min=total_dem,
        total_capacity_min=total_cap,
        utilization=(total_dem / total_cap) if total_cap else float("inf"),
        shortfall=short,
    )
    rep.suggestions = suggest(state, res) if res.shortfall else []
    return rep


def _fresh(state: State) -> State:
    t = state.clone()
    t.plan = [b for b in t.plan if b.status != "planned"]
    return t


def _all_fit(trial: State) -> bool:
    return not schedule(trial, keep_previous=False).shortfall


def suggest(state: State, res: ScheduleResult, max_items: int = 3) -> List[str]:
    """Repair options that make the WHOLE plan feasible, each verified by re-running the solver.

    1) the smallest deadline extension of a single task (prefers lower priority, then the short task)
    2) the smallest scope reduction of a single task
    3) how much extra availability is needed (an estimate, stated as such)
    """
    total_short = sum(res.shortfall.values())
    tasks = [t for t in state.tasks if t.remaining_min > 0]
    short_ids = set(res.shortfall)
    # lower priority first (3 = low), then tasks that are actually short
    tasks.sort(key=lambda t: (-t.priority, t.id not in short_ids, t.deadline))
    out: List[str] = []

    # prefer adjusting lower-priority work: try priority groups from low (3) to high (1)
    groups = sorted({t.priority for t in tasks}, reverse=True)
    best_ext = None   # (days, task, new_deadline)
    for pr in groups:
        for t in [x for x in tasks if x.priority == pr]:
            for k in range(1, 8):
                if best_ext and k >= best_ext[0]:
                    break
                trial = _fresh(state)
                tt = trial.task_by_id(t.id)
                tt.deadline = t.deadline + timedelta(days=k)
                if _all_fit(trial):
                    best_ext = (k, t, tt.deadline)
                    break
        if best_ext:
            break
    if best_ext:
        k, t, nd = best_ext
        out.append(f"'{t.name}' 마감을 {k}일 연장 ({fmt_dt(t.deadline)} → {fmt_dt(nd)})하면 모든 작업이 마감 전에 배치됩니다")

    best_cut = None   # (fraction cut, minutes, task)
    for pr in groups:
        for t in [x for x in tasks if x.priority == pr]:
            lo, hi = SLOT_MIN, t.remaining_min
            trial = _fresh(state)
            trial.task_by_id(t.id).done_min += hi
            if not _all_fit(trial):
                continue                       # even dropping this task entirely is not enough
            while lo < hi:                     # smallest cut (in 30-min steps) that makes everything fit
                mid = max(((lo + hi) // 2 // SLOT_MIN) * SLOT_MIN, lo)
                trial = _fresh(state)
                trial.task_by_id(t.id).done_min += mid
                if _all_fit(trial):
                    hi = mid
                else:
                    lo = mid + SLOT_MIN
            cand = (lo / t.remaining_min, lo, t)
            if best_cut is None or cand[:2] < best_cut[:2]:
                best_cut = cand
        if best_cut:
            break
    if best_cut:
        _, m, t = best_cut
        if m >= t.remaining_min:
            out.append(f"'{t.name}'을(를) 이번 계획에서 빼면 ({t.remaining_min/60:g}시간) 모든 작업이 배치됩니다")
        else:
            out.append(f"'{t.name}' 작업량을 {m/60:g}시간 줄이면 (남은 {t.remaining_min/60:g}시간 → {(t.remaining_min-m)/60:g}시간) 모든 작업이 배치됩니다")

    out.append(f"또는 마감 전에 작업 가능 시간을 약 {total_short/60:g}시간 더 확보해야 합니다 (예: 주말 오전 추가)")
    if not best_ext and not best_cut:
        out.insert(0, "작업 하나만 조정해서는 해결되지 않습니다. 여러 작업의 범위나 마감을 함께 조정해야 합니다")
    return out[:max_items]
