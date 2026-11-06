"""Core data structures shared by the solver, agent, app and evaluation."""
from __future__ import annotations

import copy
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

SLOT_MIN = 30  # scheduling granularity (minutes)
FMT = "%Y-%m-%d %H:%M"


def parse_dt(s: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM' (also accepts 'T' separator, seconds, '24:00', date only)."""
    s = str(s).strip().replace("T", " ")
    try:
        if len(s) >= 16:
            day, hm = s[:10], s[11:16]
            if hm == "24:00":
                return datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)
            return datetime.strptime(f"{day} {hm}", "%Y-%m-%d %H:%M")
        if len(s) == 10:
            return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        pass
    raise ValueError(f"날짜/시간 형식을 해석할 수 없습니다: {s!r} (YYYY-MM-DD HH:MM 형식 필요)")


def fmt_dt(d: datetime) -> str:
    return d.strftime(FMT)


def overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


def new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Event:
    """A busy interval from the user's calendar (read-only)."""
    start: datetime
    end: datetime
    title: str = "일정"
    id: str = field(default_factory=new_id)


@dataclass
class Blocked:
    """A user-declared unavailable interval ("내일 저녁은 안 돼")."""
    start: datetime
    end: datetime
    reason: str = ""
    id: str = field(default_factory=new_id)


@dataclass
class Task:
    name: str
    duration_min: int            # total estimated effort
    deadline: datetime
    priority: int = 2            # 1=high, 2=normal, 3=low
    max_block_min: int = 120     # longest contiguous session
    done_min: int = 0            # effort already completed
    id: str = field(default_factory=new_id)

    @property
    def remaining_min(self) -> int:
        return max(0, self.duration_min - self.done_min)


@dataclass
class Block:
    """A scheduled work session for a task."""
    task_id: str
    start: datetime
    end: datetime
    status: str = "planned"      # planned | done | missed

    @property
    def minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)

    def key(self):
        return (self.task_id, self.start, self.end)


@dataclass
class Preferences:
    # daily availability windows, (HH:MM, HH:MM); applied to every day
    windows: List[List[str]] = field(default_factory=lambda: [["09:00", "12:00"], ["13:00", "18:00"], ["19:00", "23:00"]])
    max_work_min_per_day: int = 360
    horizon_days: int = 7


# Safety: hard limits the user cannot override through chat
SLEEP_START = time(0, 0)
SLEEP_END = time(7, 0)
HARD_MAX_WORK_MIN_PER_DAY = 600


@dataclass
class State:
    now: datetime
    events: List[Event] = field(default_factory=list)
    blocked: List[Blocked] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    plan: List[Block] = field(default_factory=list)
    prefs: Preferences = field(default_factory=Preferences)
    extra_windows: List[Blocked] = field(default_factory=list)  # one-off extra availability

    def clone(self) -> "State":
        return copy.deepcopy(self)

    def task_by_id(self, tid: str) -> Optional[Task]:
        return next((t for t in self.tasks if t.id == tid), None)

    def task_by_name(self, name: str) -> Optional[Task]:
        name_n = name.strip().lower().replace(" ", "")
        exact = [t for t in self.tasks if t.name.lower().replace(" ", "") == name_n]
        if exact:
            return exact[0]
        part = [t for t in self.tasks if name_n in t.name.lower().replace(" ", "") or t.name.lower().replace(" ", "") in name_n]
        return part[0] if len(part) == 1 else None

    def day_list(self) -> List[date]:
        return [(self.now + timedelta(days=i)).date() for i in range(self.prefs.horizon_days + 1)]

    # ---- time progression -------------------------------------------------
    def advance_to(self, new_now: datetime) -> None:
        """Move the clock forward. Past planned blocks are optimistically marked done
        (the user can later say "못 했어", which reverts them)."""
        for b in self.plan:
            if b.status == "planned" and b.end <= new_now:
                b.status = "done"
                t = self.task_by_id(b.task_id)
                if t:
                    t.done_min += b.minutes
            elif b.status == "planned" and b.start < new_now < b.end:
                # partially elapsed block: credit the elapsed part, keep the rest
                t = self.task_by_id(b.task_id)
                elapsed = int((new_now - b.start).total_seconds() // 60)
                if t:
                    t.done_min += elapsed
                self.plan.append(Block(b.task_id, b.start, new_now, "done"))
                b.start = new_now
        self.now = new_now

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


def weekday_ko(d: date) -> str:
    return "월화수목금토일"[d.weekday()]
