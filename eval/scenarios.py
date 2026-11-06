"""Synthetic multi-turn scheduling scenarios with programmatically computed gold states.

Each scenario = calendar + availability + tasks (turn 0) followed by 2 natural-language
follow-up turns. For every turn we store the *gold* state (what the world really looks like
after the utterance) so plans from any system can be checked automatically.

Gold feasibility is *certified*:
  feasible   -> a constraint-valid complete plan exists (our solver found one; the checker re-verifies it)
  infeasible -> cumulative demand exceeds capacity before some deadline (a necessary condition fails)
  None       -> neither certificate (excluded from detection metrics)
"""
from __future__ import annotations

import json
import random
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional

from planner.feasibility import capacity_until
from planner.models import Blocked, Event, Preferences, State, Task, fmt_dt, parse_dt, weekday_ko
from planner.solver import schedule

NOW0 = datetime(2026, 11, 9, 9, 0)  # Monday
TASK_POOL = ["알고리즘 과제", "논문 읽기", "중간고사 공부", "보고서 작성", "발표 준비", "코딩 테스트 연습",
             "영어 회화 준비", "실험 데이터 정리", "프로젝트 기획서", "독서 감상문"]
NEW_POOL = ["포트폴리오 정리", "장학금 서류 작성", "블로그 글 작성", "동아리 회계 정리"]
TIME_TURNS = ["missed", "partial"]
OTHER_TURNS = ["block_evening", "block_range", "deadline", "add", "more", "remove"]


# ---------------------------------------------------------------------------
# (de)serialisation
# ---------------------------------------------------------------------------

def state_to_dict(s: State) -> Dict:
    return {
        "now": fmt_dt(s.now),
        "prefs": {"windows": s.prefs.windows, "max_work_min_per_day": s.prefs.max_work_min_per_day,
                  "horizon_days": s.prefs.horizon_days},
        "events": [{"start": fmt_dt(e.start), "end": fmt_dt(e.end), "title": e.title} for e in s.events],
        "blocked": [{"start": fmt_dt(b.start), "end": fmt_dt(b.end), "reason": b.reason} for b in s.blocked],
        "tasks": [{"name": t.name, "duration_min": t.duration_min, "done_min": t.done_min,
                   "deadline": fmt_dt(t.deadline), "priority": t.priority, "max_block_min": t.max_block_min}
                  for t in s.tasks],
    }


def state_from_dict(d: Dict) -> State:
    p = d["prefs"]
    return State(
        now=parse_dt(d["now"]),
        prefs=Preferences([list(w) for w in p["windows"]], p["max_work_min_per_day"], p["horizon_days"]),
        events=[Event(parse_dt(e["start"]), parse_dt(e["end"]), e.get("title", "일정")) for e in d["events"]],
        blocked=[Blocked(parse_dt(b["start"]), parse_dt(b["end"]), b.get("reason", "")) for b in d["blocked"]],
        tasks=[Task(t["name"], t["duration_min"], parse_dt(t["deadline"]), t["priority"], t["max_block_min"], t["done_min"])
               for t in d["tasks"]],
    )


def certify(s: State) -> Optional[bool]:
    g = s.clone()
    g.plan = []
    if not schedule(g, keep_previous=False).shortfall:
        return True
    demand = 0
    for t in sorted(g.tasks, key=lambda t: t.deadline):
        demand += t.remaining_min
        if demand > capacity_until(g, t.deadline):
            return False
    return None


# ---------------------------------------------------------------------------

def _j(word: str, with_batchim: str, without: str) -> str:
    """Korean particle selection (은/는, 이/가, 으로/로) by final consonant."""
    c = word[-1]
    if "가" <= c <= "힣":
        b = (ord(c) - 0xAC00) % 28
        if with_batchim == "으로" and b == 8:   # ㄹ batchim takes 로
            return word + without
        return word + (with_batchim if b else without)
    return word + without


def _day(i: int) -> datetime:
    return datetime.combine(NOW0.date() + timedelta(days=i), time(0))


def _day_phrase(i: int, rng: random.Random) -> str:
    if i == 1 and rng.random() < 0.6:
        return "내일"
    if i == 2 and rng.random() < 0.5:
        return "모레"
    return f"{weekday_ko(_day(i).date())}요일"


def _base(rng: random.Random, infeasible: bool) -> State:
    cap = rng.choice([300, 360, 360, 420])
    s = State(now=NOW0, prefs=Preferences([["09:00", "12:00"], ["13:00", "18:00"], ["19:00", "23:00"]], cap, 7))
    for i in range(7):
        for _ in range(rng.randint(1, 3)):
            st = _day(i) + timedelta(hours=rng.randint(9, 20), minutes=rng.choice([0, 30]))
            en = st + timedelta(minutes=rng.choice([60, 90, 120, 180]))
            if i == 0 and st < NOW0:
                continue
            if not any(e.start < en and st < e.end for e in s.events):
                s.events.append(Event(st, min(en, _day(i) + timedelta(hours=23)), "일정"))
    names = rng.sample(TASK_POOL, rng.randint(3, 4))
    for n in names:
        dl_day = rng.randint(2, 6)
        dl = _day(dl_day) + (timedelta(hours=23, minutes=59) if rng.random() < 0.7 else timedelta(hours=18))
        s.tasks.append(Task(n, rng.choice([2, 3, 4, 5, 6]) * 60, dl, rng.choice([1, 2, 2, 3]), 120))
    if infeasible:
        t = min(s.tasks, key=lambda t: t.deadline)
        for _ in range(40):
            if certify(s) is False and _shortfall_lb(s) >= 120:
                break
            t.duration_min += 60
    else:
        for _ in range(40):
            g = s.clone()
            total_cap = capacity_until(g, NOW0 + timedelta(days=8))
            if certify(s) is True and sum(x.remaining_min for x in s.tasks) <= 0.75 * total_cap:
                break
            big = max(s.tasks, key=lambda x: x.duration_min)
            big.duration_min = max(60, big.duration_min - 60)
    return s


def _shortfall_lb(s: State) -> int:
    demand, worst = 0, 0
    for t in sorted(s.tasks, key=lambda t: t.deadline):
        demand += t.remaining_min
        worst = max(worst, demand - capacity_until(s, t.deadline))
    return worst


def _apply_turn(kind: str, s: State, rng: random.Random) -> Optional[Dict]:
    """Mutates gold state `s`; returns turn dict (utterance etc.) or None if not applicable."""
    turn: Dict = {"type": kind}
    if kind in ("missed", "partial"):
        s.now = _day(0) + timedelta(hours=23)
        turn["advance_to"] = fmt_dt(s.now)
        if kind == "missed":
            turn["utterance"] = rng.choice(["오늘 계획했던 거 하나도 못 했어.", "오늘 일이 생겨서 계획한 작업을 전혀 못 했어.",
                                            "오늘은 아무것도 못 했어 ㅠㅠ 다시 짜줘"])
        else:
            t = rng.choice(s.tasks)
            h = rng.choice([1, 1.5, 2])
            t.done_min = min(t.duration_min, t.done_min + int(h * 60))
            turn["utterance"] = f"오늘 {_j(t.name, '은', '는')} {h:g}시간 했고 나머지는 못 했어."
        return turn
    if kind == "block_evening":
        i = rng.randint(1, 5)
        s.blocked.append(Blocked(_day(i) + timedelta(hours=18), _day(i + 1), "약속"))
        turn["utterance"] = rng.choice([f"{_day_phrase(i, rng)} 저녁엔 약속이 있어서 안 돼.",
                                        f"{_day_phrase(i, rng)} 저녁은 시간이 없어"])
        return turn
    if kind == "block_range":
        i = rng.randint(1, 5)
        a, b = rng.choice([(14, 17), (10, 12), (15, 18), (9, 11)])
        s.blocked.append(Blocked(_day(i) + timedelta(hours=a), _day(i) + timedelta(hours=b), "병원"))
        ha = f"오전 {a}시" if a < 12 else f"오후 {a-12}시"
        hb = f"{b}시" if b <= 12 else f"{b-12}시"
        turn["utterance"] = f"{_day_phrase(i, rng)} {ha}부터 {hb}까지 병원 가야 해."
        return turn
    if kind == "deadline":
        cands = [t for t in s.tasks if (t.deadline.date() - s.now.date()).days >= 3]
        if not cands:
            return None
        t = rng.choice(cands)
        i = rng.randint(2, (t.deadline.date() - NOW0.date()).days - 1)
        t.deadline = _day(i) + timedelta(hours=23, minutes=59)
        turn["utterance"] = f"{t.name} 마감이 {_j(_day_phrase(i, rng), '으로', '로')} 당겨졌어."
        return turn
    if kind == "add":
        n = rng.choice([x for x in NEW_POOL if not s.task_by_name(x)])
        i, h = rng.randint(3, 6), rng.choice([2, 3, 4])
        s.tasks.append(Task(n, h * 60, _day(i) + timedelta(hours=23, minutes=59), 2, 120))
        turn["utterance"] = f"{_day_phrase(i, rng)}까지 {n} {h}시간짜리 추가해줘."
        turn["new_task"] = n
        return turn
    if kind == "more":
        t = rng.choice(s.tasks)
        h = rng.choice([1, 2, 3])
        t.duration_min += h * 60
        turn["utterance"] = f"{t.name} 생각보다 오래 걸릴 것 같아. {h}시간 더 필요해."
        return turn
    if kind == "remove":
        t = rng.choice(s.tasks)
        s.tasks = [x for x in s.tasks if x.id != t.id]
        turn["utterance"] = f"{_j(t.name, '은', '는')} 이번 주에 안 하기로 했어. 빼줘."
        return turn
    raise ValueError(kind)


def generate(n: int = 30, seed: int = 7, infeasible_ratio: float = 0.3) -> List[Dict]:
    rng = random.Random(seed)
    out = []
    for k in range(n):
        infeasible = k < int(n * infeasible_ratio)
        s = _base(rng, infeasible)
        sc = {"id": f"S{k:02d}", "initial": state_to_dict(s), "initial_feasible": certify(s), "turns": []}
        kinds = [rng.choice(TIME_TURNS) if rng.random() < 0.5 else None, None]
        kinds = [x for x in kinds if x] or []
        while len(kinds) < 2:
            c = rng.choice(OTHER_TURNS)
            if c not in kinds:
                kinds.append(c)
        for kind in kinds:
            turn = _apply_turn(kind, s, rng)
            if turn is None:
                turn = _apply_turn("block_range", s, rng)
            turn["gold"] = state_to_dict(s)
            turn["gold_feasible"] = certify(s)
            sc["turns"].append(turn)
        out.append(sc)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="eval/scenarios.json")
    a = ap.parse_args()
    data = generate(a.n, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"{len(data)} scenarios -> {a.out}")
    print("initial feasible:", Counter(str(d["initial_feasible"]) for d in data))
    print("turn types:", Counter(t["type"] for d in data for t in d["turns"]))
    print("turn feasible:", Counter(str(t["gold_feasible"]) for d in data for t in d["turns"]))
