"""Run the evaluation: ours vs. plain-ChatGPT baselines on the same multi-turn scenarios.

  .venv/bin/python -m eval.run_eval                 # all scenarios, ours + baseline + baseline_cot
  .venv/bin/python -m eval.run_eval --limit 5       # quick smoke run
  .venv/bin/python -m eval.run_eval --judge         # + LLM-judge explanation faithfulness
  .venv/bin/python -m eval.run_eval --report-only   # rebuild summary from results/raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Dict, List, Optional

from dotenv import load_dotenv

from eval.baseline import Baseline
from eval.checker import check, stability, state_match
from eval.scenarios import state_from_dict
from planner.agent import Planner, replan
from planner.feasibility import analyze
from planner.solver import schedule
from planner.models import fmt_dt, parse_dt, weekday_ko

load_dotenv()
RESULTS = "results"


class CountingClient:
    """Wraps OpenAI client to count tokens and latency (thread-safe)."""

    def __init__(self, client):
        self._c = client
        self.lock = threading.Lock()
        self.usage = defaultdict(lambda: [0, 0, 0])   # tag -> [calls, prompt, completion]
        self.tag = threading.local()
        outer = self

        class _Comp:
            def create(self, **kw):
                r = outer._c.chat.completions.create(**kw)
                tag = getattr(outer.tag, "v", "other")
                with outer.lock:
                    u = outer.usage[tag]
                    u[0] += 1
                    if r.usage:
                        u[1] += r.usage.prompt_tokens
                        u[2] += r.usage.completion_tokens
                return r

        class _Chat:
            completions = _Comp()

        self.chat = _Chat()
        self.moderations = client.moderations


def export_plan(state) -> List[Dict]:
    return [{"task": state.task_by_id(b.task_id).name if state.task_by_id(b.task_id) else "?",
             "start": fmt_dt(b.start), "end": fmt_dt(b.end)} for b in state.plan if b.status == "planned"]


def own_infeasible(state) -> bool:
    fut = [b for b in state.plan if b.status == "planned"]
    return any(sum(b.minutes for b in fut if b.task_id == t.id) < t.remaining_min for t in state.tasks)


def run_ours(sc: Dict, client, model: str) -> List[Dict]:
    client.tag.v = "ours"
    state = state_from_dict(sc["initial"])
    replan(state)
    p = Planner(state, client=client, model=model, use_moderation=False)
    out = [{"turn": 0, "type": "initial", "plan": export_plan(state), "infeasible": own_infeasible(state),
            "explanation": "", "latency": 0.0}]
    for i, t in enumerate(sc["turns"], 1):
        if t.get("advance_to"):
            state.advance_to(parse_dt(t["advance_to"]))
        t0 = time.time()
        tr = p.chat(t["utterance"])
        gold = state_from_dict(t["gold"])
        out.append({"turn": i, "type": t["type"], "plan": export_plan(state), "infeasible": own_infeasible(state),
                    "explanation": tr.reply, "latency": time.time() - t0,
                    "ops": [{"name": o["name"], "args": o["args"], "ok": o["ok"], "msg": o["msg"]} for o in tr.ops],
                    "state_match": state_match(gold, state)})
    return out


def run_baseline(sc: Dict, client, model: str, cot: bool) -> List[Dict]:
    client.tag.v = "baseline_cot" if cot else "baseline"
    b = Baseline(client, model, cot)
    init = state_from_dict(sc["initial"])
    t0 = time.time()
    r = b.start(init)
    out = [{"turn": 0, "type": "initial", "plan": r["blocks"], "infeasible": bool(r.get("infeasible")),
            "explanation": r.get("explanation", ""), "latency": time.time() - t0, "parse_error": r.get("_parse_error", False)}]
    for i, t in enumerate(sc["turns"], 1):
        g = state_from_dict(t["gold"])
        t0 = time.time()
        r = b.follow_up(fmt_dt(g.now), weekday_ko(g.now.date()), t["utterance"])
        out.append({"turn": i, "type": t["type"], "plan": r["blocks"], "infeasible": bool(r.get("infeasible")),
                    "explanation": r.get("explanation", ""), "latency": time.time() - t0, "parse_error": r.get("_parse_error", False)})
    return out


JUDGE = """당신은 일정 챗봇의 설명이 실제 계획 변경과 일치하는지 검증하는 평가자입니다.
[사용자 발화]와 [실제 계획 변화](시스템이 출력한 이전/이후 계획을 비교해 기계적으로 계산한 것), [실제 상태]가 주어집니다.
[챗봇 설명]에서 일정·시간·작업량·마감·실현가능성에 관한 구체적 사실 주장을 모두 뽑고, 각각 실제 정보로 뒷받침되는지 판정하세요.
JSON으로만 답하세요: {"claims": [{"claim": "...", "supported": true/false}]}"""


def judge_faithfulness(client, model: str, utterance: str, gold, prev_plan, new_plan, explanation: str) -> Optional[float]:
    if not explanation.strip():
        return None
    def keyset(p):
        return {(b["task"], b["start"], b["end"]) for b in p}
    removed = sorted(keyset(prev_plan) - keyset(new_plan), key=lambda x: x[1])
    added = sorted(keyset(new_plan) - keyset(prev_plan), key=lambda x: x[1])
    res = check(gold, new_plan)
    # ground-truth workload facts computed from the gold state (identical for every system)
    g = gold.clone()
    g.plan = []
    sres = schedule(g, keep_previous=False)
    rep = analyze(g, sres)
    facts = (f"빠진 블록: {removed[:25]}\n추가된 블록: {added[:25]}\n"
             f"변경 후 전체 계획: {sorted(keyset(new_plan), key=lambda x: x[1])[:60]}\n"
             f"이 계획의 제약 위반: {res['violations']}, 하루 최대시간 초과 일수 {res['cap_exceeded_days']}, "
             f"남은 작업 커버율 {res['coverage']*100:.0f}%\n"
             f"작업별 남은 시간/마감: {[(t.name, t.remaining_min/60, fmt_dt(t.deadline)) for t in gold.tasks]}\n"
             f"[정답 사실] 총 남은 작업 {rep.total_demand_min/60:g}시간, 총 가용 {rep.total_capacity_min/60:g}시간, "
             f"실현 가능={rep.feasible}, 최적 배치로도 부족한 시간={ {k: v/60 for k, v in rep.shortfall.items()} }\n"
             f"[정답 사실] 사용자가 막은 시간: {[(fmt_dt(b.start), fmt_dt(b.end), b.reason) for b in gold.blocked]}\n"
             f"[정답 사실] 부하율(남은 작업/가용) = {rep.utilization*100:.0f}%\n"
             f"[정답 사실] 마감별 누적 필요/가용: {[(r.deadline, r.demand_min/60, r.capacity_min/60) for r in rep.rows]}\n"
             f"[정답 사실] 솔버로 검증된 해결책: {rep.suggestions}")
    client.tag.v = "judge"
    r = client.chat.completions.create(model=model, temperature=0, response_format={"type": "json_object"}, messages=[
        {"role": "system", "content": JUDGE},
        {"role": "user", "content": f"[사용자 발화]\n{utterance}\n\n[실제 계획 변화]\n{facts}\n\n[챗봇 설명]\n{explanation}"}])
    try:
        claims = json.loads(r.choices[0].message.content).get("claims", [])
    except Exception:  # noqa: BLE001
        return None
    if not claims:
        return None
    return sum(1 for c in claims if c.get("supported")) / len(claims)


def evaluate_scenario(sc: Dict, client, model: str, systems: List[str], judge: bool, judge_model: str) -> List[Dict]:
    rows = []
    for sysname in systems:
        try:
            if sysname == "ours":
                turns = run_ours(sc, client, model)
            else:
                turns = run_baseline(sc, client, model, cot=(sysname == "baseline_cot"))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rows.append({"scenario": sc["id"], "system": sysname, "error": str(e)})
            continue
        prev_plan = None
        for t in turns:
            if t["turn"] == 0:
                gold, gfeas, utt = state_from_dict(sc["initial"]), sc["initial_feasible"], ""
            else:
                td = sc["turns"][t["turn"] - 1]
                gold, gfeas, utt = state_from_dict(td["gold"]), td["gold_feasible"], td["utterance"]
            row = {"scenario": sc["id"], "system": sysname, **{k: v for k, v in t.items() if k != "plan"},
                   "gold_feasible": gfeas, "utterance": utt, "check": check(gold, t["plan"]), "plan": t["plan"]}
            if prev_plan is not None:
                row["stability"] = stability(gold, prev_plan, t["plan"])
                if judge:
                    row["faithfulness"] = judge_faithfulness(client, judge_model, utt, gold, prev_plan, t["plan"], t["explanation"])
            prev_plan = t["plan"]
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _avg(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else float("nan")


def _prf(rows):
    tp = sum(1 for r in rows if r["gold_feasible"] is False and r["infeasible"])
    fp = sum(1 for r in rows if r["gold_feasible"] is True and r["infeasible"])
    fn = sum(1 for r in rows if r["gold_feasible"] is False and not r["infeasible"])
    tn = sum(1 for r in rows if r["gold_feasible"] is True and not r["infeasible"])
    p = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * rc / (p + rc) if p == p and rc == rc and p + rc else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else float("nan")
    return p, rc, f1, acc


def _pct(v):
    return "-" if v != v else f"{v*100:.1f}%"


def feature_section(L: List[str], rows: List[Dict], systems: List[str]) -> None:
    """Main results grouped by the three features; each metric belongs to exactly one feature."""
    def by(s, turn_filter):
        return [r for r in rows if r["system"] == s and turn_filter(r)]

    init = lambda r: r["turn"] == 0          # noqa: E731
    follow = lambda r: r["turn"] > 0         # noqa: E731
    allt = lambda r: True                    # noqa: E731
    hard = lambda rs: _avg([1.0 if r["check"]["hard_ok"] else 0.0 for r in rs])                      # noqa: E731
    conflict = lambda rs: _avg([r["check"]["conflict_rate"] for r in rs])                             # noqa: E731
    deadline = lambda rs: _avg([r["check"]["deadline_violation_rate"] for r in rs])                   # noqa: E731
    complete = lambda rs: _avg([1.0 if r["check"]["complete_valid"] else 0.0 for r in rs if r["gold_feasible"] is True])  # noqa: E731

    def parse_acc(rs):
        rs = [r for r in rs if "state_match" in r]
        return _avg([1.0 if r["state_match"]["ok"] else 0.0 for r in rs]) if rs else float("nan")

    groups = [
        ("① 제약을 지키는 계획 생성", "초기 계획 (turn 0)", init, [
            ("제약 완전 준수 계획 비율 ↑", hard),
            ("일정 충돌 블록 비율 ↓", conflict),
            ("마감 위반 블록 비율 ↓", deadline),
            ("완전·유효 계획 비율 (실현가능 케이스, 커버율 포함) ↑", complete),
        ]),
        ("② 상황 변화에 따른 재조정", "후속 발화 (turn 1–2)", follow, [
            ("발화 → 상태 파싱 정확도 ↑", parse_acc),
            ("재조정 후 제약 완전 준수 비율 ↑", hard),
            ("기존 블록 유지율 (안정성) ↑", lambda rs: _avg([r.get("stability") for r in rs])),
        ]),
        ("③ 실현 가능성 진단과 설명", "탐지: 모든 턴 · 설명: 후속 발화", allt, [
            ("불가능 탐지 F1 ↑", lambda rs: _prf([r for r in rs if r["gold_feasible"] is not None])[2]),
            ("불가능 탐지 Precision / Recall", None),
            ("설명 사실 일치율 (LLM judge) ↑", lambda rs: _avg([r.get("faithfulness") for r in rs if r["turn"] > 0])),
        ]),
    ]
    L.append("\n# 기능별 결과\n")
    for title, scope, tf, metrics in groups:
        L.append(f"\n## {title}\n\n측정 대상: {scope}\n")
        L.append("| 지표 | " + " | ".join(systems) + " |")
        L.append("|---|" + "---|" * len(systems))
        for name, fn in metrics:
            vals = []
            for s in systems:
                rs = by(s, tf)
                if fn is None:   # precision / recall pair
                    p, rc, _, _ = _prf([r for r in rs if r["gold_feasible"] is not None])
                    vals.append(f"{_pct(p)} / {_pct(rc)}")
                else:
                    v = fn(rs)
                    vals.append("해당 없음" if v != v and "파싱" in name and s != "ours" else _pct(v))
            L.append(f"| {name} | " + " | ".join(vals) + " |")
    L.append("\n## 별도 평가\n")
    L.append("- 안전성 red-team: `results/safety_summary.md` (루브릭 Safety 항목용)")
    L.append("- 보조 지표(응답 지연, 토큰 비용), 세부 위반 항목(하루 최대시간 초과, 과잉 배치): 아래 부록")


def summarize(rows: List[Dict], usage: Optional[Dict] = None) -> str:
    rows = [r for r in rows if "error" not in r]
    systems = [s for s in ["ours", "baseline", "baseline_cot"] if any(r["system"] == s for r in rows)]
    L = ["# 평가 결과\n"]
    n_sc = len({r["scenario"] for r in rows})
    L.append(f"시나리오 {n_sc}개 × (초기 계획 + 후속 발화 2턴). 모든 지표는 **gold 상태** 기준으로 자동 채점.\n")
    feature_section(L, rows, systems)
    L.append("\n---\n\n# 부록: 세부 지표\n")

    def table(title, subset_fn):
        L.append(f"\n## {title}\n")
        L.append("| 지표 | " + " | ".join(systems) + " |")
        L.append("|---|" + "---|" * len(systems))
        metrics = [
            ("제약 완전 준수 계획 비율 ↑", lambda rs: _avg([1.0 if r["check"]["hard_ok"] else 0.0 for r in rs])),
            ("일정 충돌 블록 비율 ↓", lambda rs: _avg([r["check"]["conflict_rate"] for r in rs])),
            ("마감 위반 블록 비율 ↓", lambda rs: _avg([r["check"]["deadline_violation_rate"] for r in rs])),
            ("하루 최대시간 초과 일수(평균) ↓", lambda rs: _avg([r["check"]["cap_exceeded_days"] for r in rs])),
            ("필요 작업시간 커버율(실현가능 케이스) ↑", lambda rs: _avg([r["check"]["coverage"] for r in rs if r["gold_feasible"] is True])),
            ("완전·유효 계획 비율(실현가능 케이스) ↑", lambda rs: _avg([1.0 if r["check"]["complete_valid"] else 0.0 for r in rs if r["gold_feasible"] is True])),
            ("과잉 배치(분, 평균) ↓", lambda rs: _avg([r["check"]["over_alloc_min"] for r in rs])),
        ]
        for name, fn in metrics:
            vals = []
            for s in systems:
                rs = subset_fn([r for r in rows if r["system"] == s])
                v = fn(rs) if rs else float("nan")
                vals.append("-" if v != v else (f"{v*100:.1f}%" if "비율" in name or "커버율" in name else f"{v:.2f}"))
            L.append(f"| {name} | " + " | ".join(vals) + " |")

    table("A. 제약 지표: 전체 (모든 턴)", lambda rs: rs)
    table("B. 제약 지표: 초기 계획 (turn 0)", lambda rs: [r for r in rs if r["turn"] == 0])
    table("C. 제약 지표: 재조정 (후속 발화 turn 1-2)", lambda rs: [r for r in rs if r["turn"] > 0])

    L.append("\n## 재조정 품질\n")
    L.append("| 지표 | " + " | ".join(systems) + " |")
    L.append("|---|" + "---|" * len(systems))
    for name, fn in [
        ("계획 안정성 (유효한 기존 블록 유지율) ↑", lambda rs: _avg([r.get("stability") for r in rs])),
        ("설명 사실 일치율 (LLM judge) ↑", lambda rs: _avg([r.get("faithfulness") for r in rs])),
        ("응답 지연(초, 평균)", lambda rs: _avg([r.get("latency") for r in rs])),
    ]:
        vals = []
        for s in systems:
            v = fn([r for r in rows if r["system"] == s and r["turn"] > 0])
            vals.append("-" if v != v else (f"{v:.2f}" if "지연" in name else f"{v*100:.1f}%"))
        L.append(f"| {name} | " + " | ".join(vals) + " |")

    L.append("\n## 실현 불가능 탐지 (positive = 불가능)\n")
    L.append("| 시스템 | Precision | Recall | F1 | Accuracy |")
    L.append("|---|---|---|---|---|")
    for s in systems:
        p, rc, f1, acc = _prf([r for r in rows if r["system"] == s and r["gold_feasible"] is not None])
        L.append(f"| {s} | {p*100:.1f}% | {rc*100:.1f}% | {f1*100:.1f}% | {acc*100:.1f}% |")

    L.append("\n## 후속 발화 유형별 제약 완전 준수 비율\n")
    types = sorted({r["type"] for r in rows if r["turn"] > 0})
    L.append("| 유형 | n | " + " | ".join(systems) + " |")
    L.append("|---|---|" + "---|" * len(systems))
    for ty in types:
        vals = []
        n = 0
        for s in systems:
            rs = [r for r in rows if r["system"] == s and r["type"] == ty]
            n = len(rs)
            vals.append(f"{_avg([1.0 if r['check']['hard_ok'] else 0.0 for r in rs])*100:.0f}%")
        L.append(f"| {ty} | {n} | " + " | ".join(vals) + " |")

    ours = [r for r in rows if r["system"] == "ours" and r["turn"] > 0]
    if ours:
        acc = _avg([1.0 if r["state_match"]["ok"] else 0.0 for r in ours])
        L.append(f"\n## 발화 → 상태 변경 파싱 정확도 (ours)\n\n상태가 gold와 정확히 일치한 턴: **{acc*100:.1f}%** ({sum(r['state_match']['ok'] for r in ours)}/{len(ours)})\n")
        L.append("| 유형 | 정확도 |")
        L.append("|---|---|")
        for ty in types:
            rs = [r for r in ours if r["type"] == ty]
            L.append(f"| {ty} | {_avg([1.0 if r['state_match']['ok'] else 0.0 for r in rs])*100:.0f}% ({sum(r['state_match']['ok'] for r in rs)}/{len(rs)}) |")

    L.append("\n## 오류 분석 (위반 유형별 블록 수 합계)\n")
    L.append("| 위반 유형 | " + " | ".join(systems) + " |")
    L.append("|---|" + "---|" * len(systems))
    kinds = ["conflict", "after_deadline", "outside_window", "in_past", "self_overlap", "unknown_task", "malformed"]
    for k in kinds:
        L.append(f"| {k} | " + " | ".join(str(sum(r["check"]["violations"].get(k, 0) for r in rows if r["system"] == s)) for s in systems) + " |")
    L.append("| cap_exceeded_days | " + " | ".join(str(sum(r["check"]["cap_exceeded_days"] for r in rows if r["system"] == s)) for s in systems) + " |")

    if ours:
        errs = Counter(e.split(":")[0] for r in ours for e in r["state_match"]["errors"])
        L.append("\n### ours 파싱 오류 유형\n")
        L.append(", ".join(f"{k}: {v}" for k, v in errs.most_common()) or "없음")
        L.append("\n### ours 실패 사례\n")
        for r in [r for r in ours if not r["state_match"]["ok"]][:12]:
            L.append(f"- **{r['scenario']} ({r['type']})** “{r['utterance']}” → ops={[(o['name'], o['args']) for o in r.get('ops', [])]} / 오류={r['state_match']['errors']}")
        fails = [r for r in ours if not r["check"]["hard_ok"]]
        L.append(f"\n제약 위반이 발생한 ours 턴: {len(fails)}개 " + (str([(r['scenario'], r['type'], r['check']['violations']) for r in fails[:10]]) if fails else ""))

    if usage:
        L.append("\n## 비용 (토큰 사용량)\n")
        L.append("| 구분 | 호출 수 | 입력 토큰 | 출력 토큰 |")
        L.append("|---|---|---|---|")
        for k, (c, p, o) in usage.items():
            L.append(f"| {k} | {c} | {p} | {o} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="eval/scenarios.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--systems", default="ours,baseline,baseline_cot")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--rejudge", action="store_true", help="recompute faithfulness on results/raw.jsonl only")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    raw_path = os.path.join(RESULTS, "raw.jsonl")

    if a.rejudge:
        from openai import OpenAI
        client = CountingClient(OpenAI())
        rows = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
        scs = {sc["id"]: sc for sc in json.load(open(a.scenarios, encoding="utf-8"))}
        prev = {(r["scenario"], r["system"], r["turn"]): r["plan"] for r in rows if "plan" in r}
        todo = [r for r in rows if r.get("turn", 0) > 0 and "plan" in r]

        def rj(r):
            client.tag.v = "judge"
            td = scs[r["scenario"]]["turns"][r["turn"] - 1]
            r["faithfulness"] = judge_faithfulness(client, a.judge_model, td["utterance"], state_from_dict(td["gold"]),
                                                   prev[(r["scenario"], r["system"], r["turn"] - 1)], r["plan"], r["explanation"])
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(rj, todo))
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        usage = json.load(open(os.path.join(RESULTS, "usage.json")))
        usage["judge(rejudge)"] = client.usage["judge"]
        json.dump(usage, open(os.path.join(RESULTS, "usage.json"), "w"))
    elif a.report_only:
        rows = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
        up = os.path.join(RESULTS, "usage.json")
        usage = json.load(open(up)) if os.path.exists(up) else None
    else:
        from openai import OpenAI
        client = CountingClient(OpenAI())
        scs = json.load(open(a.scenarios, encoding="utf-8"))
        if a.limit:
            scs = scs[:a.limit]
        systems = a.systems.split(",")
        rows: List[Dict] = []
        done = 0
        with ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(evaluate_scenario, sc, client, a.model, systems, a.judge, a.judge_model) for sc in scs]
            for f in futs:
                rows += f.result()
                done += 1
                print(f"\r{done}/{len(scs)} scenarios", end="", flush=True)
        print()
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        usage = dict(client.usage)
        json.dump(usage, open(os.path.join(RESULTS, "usage.json"), "w"))

    md = summarize(rows, usage)
    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
