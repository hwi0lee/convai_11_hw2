"""Streamlit UI for the calendar-grounded scheduling coach.

Run:  .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, time, timedelta

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from eval.baseline import Baseline
from eval.checker import parse_plan
from planner.agent import DEFAULT_MODEL, Planner, fmt_block, replan
from planner.calendar_source import initial_state, load_google, load_mock
from planner.feasibility import analyze
from planner.models import fmt_dt, weekday_ko
from planner.ops import apply_op
from planner.solver import ScheduleResult

load_dotenv()
st.set_page_config(page_title="일정 코치", page_icon="📅", layout="wide")

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "sample.json")
PALETTE = ["#9aa0a6", "#e8710a", "#4285f4", "#34a853", "#a142f4", "#f9ab00", "#24c1e0", "#e52592",
           "#5f6368", "#7cb342", "#8d6e63", "#00897b"]


def default_start():
    """The sample calendar is designed for a Monday start: today if Monday, else the coming Monday."""
    d = datetime.now().date()
    return d + timedelta(days=(7 - d.weekday()) % 7)


def boot(source: str, now: datetime, model: str, send_titles: bool, moderation: bool, with_tasks: bool = True):
    events, tasks = load_mock(SAMPLE, now, with_tasks=with_tasks)
    if source == "Google Calendar":
        try:
            events = load_google(now)
            st.toast(f"Google Calendar에서 일정 {len(events)}개를 불러왔습니다 (읽기 전용).")
        except Exception as e:  # noqa: BLE001
            st.error(f"Google Calendar 연결 실패: {e}\n샘플 캘린더로 대체합니다. (credentials.json 필요, README 참고)")
    state = initial_state(now, events, tasks)
    st.session_state.boot_state = state.clone()   # what the independent ChatGPT service starts from
    res, diff, rep, log = replan(state)
    st.session_state.planner = Planner(state, model=model, send_titles=send_titles, use_moderation=moderation)
    st.session_state.last_log = log
    st.session_state.baseline = None
    st.session_state.base_resp = None
    st.session_state.base_messages = []
    if not tasks:
        greeting = ("안녕하세요! 캘린더 일정을 불러왔어요. 이번 주에 해야 할 일을 마감과 예상 시간과 함께 알려 주세요.\n\n"
                    "예: *“수요일까지 NLP 과제 8시간짜리 추가해줘”*, *“목요일 6시까지 논문 리뷰 4시간”*, "
                    "*“토요일 아침 9시 시험이라 ML 공부 10시간 필요해”*")
    st.session_state.messages = [{
        "role": "assistant", "who": "ours",
        "content": greeting if not tasks else (f"안녕하세요! 캘린더와 작업 {len(tasks)}개를 바탕으로 계획을 세웠어요. "
                    f"남은 작업 {rep.total_demand_min/60:g}시간 / 가용 {rep.total_capacity_min/60:g}시간 "
                    f"(부하율 {rep.utilization*100:.0f}%) — "
                    + ("모두 마감 전에 배치됩니다." if rep.feasible else "⚠️ 일부 작업은 시간이 부족합니다.")
                    + "\n\n상황이 바뀌면 편하게 말해 주세요. 예: *“오늘 계획한 거 못 했어”*, *“내일 저녁은 시간 없어”*, "
                      "*“금요일까지 발표 준비 3시간 추가해줘”*"),
        "log": log,
    }]
    if not tasks:   # nothing to plan yet: the baseline starts on the first message
        st.session_state.base_messages = [{"role": "assistant", "content": "이번 주에 해야 할 일을 알려 주세요. 캘린더 일정을 보고 계획을 세워 드릴게요."}]
        return
    try:
        start_baseline(state, model)
    except Exception as e:  # noqa: BLE001
        st.session_state.base_messages = [{"role": "assistant", "content": f"⚠️ 기본 ChatGPT 호출 실패: {e}"}]


def md_safe(text: str) -> str:
    """LLM replies often write time ranges as 09:00~10:00; a pair of '~' renders as strikethrough in markdown."""
    return (text or "").replace("~", "\\~")


def start_baseline(state, model):
    from openai import OpenAI
    b = Baseline(OpenAI(), model)
    resp = b.start(state)
    st.session_state.baseline, st.session_state.base_resp = b, resp
    st.session_state.base_messages = [{"role": "assistant", "content": resp.get("explanation", "")}]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY가 없습니다 (.env 확인)")
    model = st.selectbox("모델", [DEFAULT_MODEL, "gpt-4o-mini", "gpt-4.1", "gpt-5-mini"], index=0)
    source = st.radio("캘린더", ["샘플 캘린더", "Google Calendar"], horizontal=True)
    send_titles = st.checkbox("일정 제목을 LLM에 전송", value=False,
                              help="기본값은 꺼짐: LLM에는 바쁜 시간대(free/busy)만 전달되어 일정 내용이 외부로 나가지 않습니다.")
    moderation = st.checkbox("입력 모더레이션 사용", value=True)
    with_tasks = st.checkbox("샘플 작업 4개 불러오기", value=False,
                             help="끄면 캘린더 일정만 있고 할 일은 비어 있는 상태로 시작합니다. 채팅으로 할 일을 추가하세요.")
    start_date = st.date_input("시작 시각(날짜)", value=default_start())
    start_time = st.time_input("시작 시각(시간)", value=time(9, 0), step=1800)
    if start_date.weekday() != 0 and source == "샘플 캘린더":
        st.caption("⚠️ 샘플 캘린더는 월요일 시작 기준으로 만들어졌습니다.")
    if st.button("🔄 불러오기 / 초기화", width="stretch") or "planner" not in st.session_state:
        boot(source, datetime.combine(start_date, start_time), model, send_titles, moderation, with_tasks)

    planner: Planner = st.session_state.planner
    planner.model, planner.send_titles, planner.use_moderation = model, send_titles, moderation
    state = planner.state

    st.divider()
    st.subheader("⏩ 시뮬레이션 시계")
    st.caption(f"현재: **{state.now:%m/%d}({weekday_ko(state.now.date())}) {state.now:%H:%M}**")
    c1, c2 = st.columns(2)
    if c1.button("+3시간"):
        state.advance_to(state.now + timedelta(hours=3))
        st.session_state.last_log = replan(state)[3]
    if c2.button("오늘 밤 23시"):
        tgt = datetime.combine(state.now.date(), time(23, 0))
        if tgt > state.now:
            state.advance_to(tgt)
            st.session_state.last_log = replan(state)[3]
    st.caption("시계는 두 서비스 공통입니다. 일정 코치는 지나간 블록을 '완료'로 간주하고, 기본 ChatGPT는 다음 메시지에서 현재 시각을 전달받습니다.")

    st.divider()
    st.subheader("🕘 작업 선호")
    cap = st.slider("하루 최대 작업시간", 1.0, 10.0, state.prefs.max_work_min_per_day / 60, 0.5)
    win = st.text_input("작업 가능 시간대", ", ".join(f"{a}-{b}" for a, b in state.prefs.windows))
    if st.button("선호 적용"):
        wins = [w.strip().split("-") for w in win.split(",") if "-" in w]
        ok, msg = apply_op(state, "set_preferences", {"max_hours_per_day": cap, "windows": wins})
        (st.success if ok else st.error)(msg)
        st.session_state.last_log = replan(state)[3]

state = st.session_state.planner.state

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def seg(label: str, options, key: str) -> str:
    """Tab-like switch that renders ONLY the selected view. (Charts drawn inside hidden st.tabs can
    finish rendering at zero width and stay blank when the tab is shown again.)"""
    v = st.segmented_control(label, options, default=options[0], key=key, label_visibility="collapsed")
    return v or options[0]


# ---------------------------------------------------------------------------
# Shared chart helpers
# ---------------------------------------------------------------------------

def frame(state, blocks, show_blocked: bool = True) -> pd.DataFrame:
    """blocks: list of (task name, start, end, status)."""
    rows = []
    days = state.day_list()

    def add(start, end, kind, label, status=""):
        cur = start
        while cur < end:   # split across midnight
            day_end = datetime.combine(cur.date() + timedelta(days=1), time(0))
            seg_end = min(end, day_end)
            if cur.date() in days:
                rows.append({"day": f"{cur:%m/%d}({weekday_ko(cur.date())})", "start": cur.hour + cur.minute / 60,
                             "end": (seg_end - datetime.combine(cur.date(), time(0))).total_seconds() / 3600,
                             "kind": kind, "label": label, "status": status,
                             "time": f"{cur:%H:%M}-{seg_end:%H:%M}"})
            cur = seg_end

    for e in state.events:
        add(e.start, e.end, "캘린더 일정", e.title)
    if show_blocked:
        for b in state.blocked:
            add(b.start, b.end, "막은 시간", b.reason or "불가")
    for kind, s, e, status in blocks:
        add(s, e, kind, kind, status)
    return pd.DataFrame(rows)


def chart(df: pd.DataFrame, state, names, key: str, height=600):
    order = [f"{d:%m/%d}({weekday_ko(d)})" for d in state.day_list()]
    domain = ["캘린더 일정", "막은 시간"] + [n for n in dict.fromkeys(names) if n not in ("캘린더 일정", "막은 시간")]
    df = df[df["status"] != "missed"].copy()
    df["chart"] = key   # makes each chart's dataset unique: identical datasets in two charts share a name in the browser
    # show 0-7h only when something is scheduled there (e.g. an all-nighter plan)
    top = 0 if (not df.empty and df["start"].min() < 7) else 7
    yscale = alt.Scale(domain=[24, top], nice=False)
    base = alt.Chart(df).encode(
        x=alt.X("day:O", sort=order, title=None, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("start:Q", scale=yscale, title="시각", axis=alt.Axis(values=list(range(top, 25, 2 if top == 0 else 1)))),
        y2="end:Q",
    )
    color = alt.Color("kind:N", scale=alt.Scale(domain=domain, range=(PALETTE * 3)[:len(domain)]),
                      legend=alt.Legend(orient="bottom", title=None, columns=3))
    tip = ["day", "time", "label", "status"]
    fade = alt.condition(alt.datum.status == "done", alt.value(0.35), alt.value(0.95))   # no scale: an empty layer cannot break it
    is_busy = alt.FieldOneOfPredicate(field="kind", oneOf=["캘린더 일정", "막은 시간"])
    # busy time drawn full width underneath; work blocks narrower on top, so an overlap stays visible
    busy = base.transform_filter(is_busy).mark_bar(cornerRadius=3, width=alt.RelativeBandSize(0.9)).encode(
        color=color, opacity=fade, tooltip=tip)
    work = base.transform_filter(~is_busy).mark_bar(cornerRadius=3, width=alt.RelativeBandSize(0.55),
                                                     stroke="white", strokeWidth=0.5).encode(
        color=color, opacity=fade, tooltip=tip)
    bars = busy + work
    if top == 0:   # shade protected sleep hours so blocks placed there stand out
        sleep = alt.Chart(pd.DataFrame([{"s": 0, "e": 7, "label": "수면 시간 (00–07시)", "chart": key}])).mark_rect(
            color="#7b61ff", opacity=0.10).encode(y=alt.Y("s:Q", scale=yscale), y2="e:Q")
        sleep_txt = alt.Chart(pd.DataFrame([{"y": 3.5, "label": "수면 시간 (00–07시)", "chart": key}])).mark_text(
            align="left", dx=4, color="#7b61ff", fontSize=11).encode(y=alt.Y("y:Q", scale=yscale), text="label:N", x=alt.value(0))
        bars = sleep + sleep_txt + bars
    now_df = pd.DataFrame([{"day": f"{state.now:%m/%d}({weekday_ko(state.now.date())})",
                            "y": state.now.hour + state.now.minute / 60, "chart": key}])
    now_rule = alt.Chart(now_df).mark_tick(color="red", thickness=2, size=60).encode(x=alt.X("day:O", sort=order), y=alt.Y("y:Q", scale=yscale))
    # a data fingerprint in the spec makes the browser rebuild the chart on every change instead of patching it in place
    stamp = hashlib.md5(df.to_json().encode()).hexdigest()[:10]
    st.altair_chart((bars + now_rule).properties(height=height, usermeta={"chart": key, "data": stamp}),
                    use_container_width=True, key=key)


# ---------------------------------------------------------------------------
# Main layout: two independent services in top-level tabs
# ---------------------------------------------------------------------------
st.title("📅 일정 코치")
st.caption("같은 캘린더·작업·시계에서 두 서비스를 각각 써 보고 비교해 보세요.")
svc = seg("서비스", ["📅 일정 코치", "💬 기본 ChatGPT"], key="svc")

# ============================ 일정 코치 =====================================
if svc == "📅 일정 코치":
    st.caption("LLM은 말을 이해하고 설명만 합니다 · 배치는 제약 솔버가 마감·충돌·가용시간을 보장합니다")
    left, right = st.columns([5, 7], gap="large")
    with left:
        box = st.container(height=640)
        for m in st.session_state.messages:
            with box.chat_message(m["role"], avatar="📅" if m["role"] == "assistant" else None):
                st.markdown(md_safe(m["content"]))
                if m.get("ops"):
                    st.caption("🛠 " + " / ".join(("✅ " if o["ok"] else "❌ ") + o["msg"] for o in m["ops"]))
                if m.get("log"):
                    with st.expander("스케줄러 로그 (설명의 근거)"):
                        st.code(m["log"], language=None)
        if prompt := st.chat_input("예: 오늘 계획한 일을 못 했어 / 내일 저녁은 시간이 없어", key="ours_input"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("생각 중..."):
                try:
                    tr = st.session_state.planner.chat(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": tr.reply,
                                                      "ops": tr.ops, "log": tr.log})
                    if tr.log:
                        st.session_state.last_log = tr.log
                except Exception as e:  # noqa: BLE001
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ 오류: {e}"})
            st.rerun()

    with right:
        view = seg("보기", ["📅 주간 계획", "📊 부하·실현 가능성", "🧾 변경 로그"], key="ours_view")
        ours_blocks = [((state.task_by_id(b.task_id).name if state.task_by_id(b.task_id) else "?"), b.start, b.end, b.status)
                       for b in state.plan]
        if view == "📅 주간 계획":
            df = frame(state, ours_blocks)
            if not df.empty:
                chart(df, state, [t.name for t in state.tasks], key="chart_ours")
                st.caption("흐린 블록 = 지나간(완료로 간주) 블록 · 빨간 선 = 현재 시각")

        if view == "📊 부하·실현 가능성":
            future = [b for b in state.plan if b.status == "planned"]
            short = {}
            for t in state.tasks:
                placed = sum(b.minutes for b in future if b.task_id == t.id)
                if placed < t.remaining_min:
                    short[t.id] = t.remaining_min - placed
            rep = analyze(state, ScheduleResult(plan=state.plan, shortfall=short))
            m1, m2, m3 = st.columns(3)
            m1.metric("남은 작업", f"{rep.total_demand_min/60:g}h")
            m2.metric("가용 시간", f"{rep.total_capacity_min/60:g}h")
            m3.metric("부하율", f"{rep.utilization*100:.0f}%" if rep.total_capacity_min else "∞")
            if rep.feasible:
                st.success("모든 작업이 마감 전에 배치되었습니다.")
            else:
                st.error("시간이 부족합니다: " + ", ".join(f"{k} {v/60:g}시간" for k, v in rep.shortfall.items()))
                st.markdown("**검증된 대안**\n" + "\n".join(f"- {s}" for s in rep.suggestions))
            st.markdown("**마감별 누적 필요시간 vs 가용시간**")
            st.dataframe(pd.DataFrame([{"마감": r.deadline, "작업": ", ".join(r.tasks), "누적 필요(h)": r.demand_min / 60,
                                        "가용(h)": r.capacity_min / 60, "부족(h)": r.shortfall_min / 60} for r in rep.rows]),
                         hide_index=True, width="stretch")
            st.markdown("**작업 목록**")
            st.dataframe(pd.DataFrame([{"작업": t.name, "남은(h)": t.remaining_min / 60, "총(h)": t.duration_min / 60,
                                        "마감": f"{t.deadline:%m/%d %H:%M}", "우선순위": t.priority,
                                        "배치됨(h)": sum(b.minutes for b in future if b.task_id == t.id) / 60}
                                       for t in sorted(state.tasks, key=lambda t: t.deadline)]),
                         hide_index=True, width="stretch")
            with st.expander("➕ 작업 직접 추가 (LLM 없이)"):
                with st.form("add"):
                    n = st.text_input("작업 이름")
                    h = st.number_input("예상 소요시간(h)", 0.5, 50.0, 2.0, 0.5)
                    dd = st.date_input("마감 날짜", value=state.now.date() + timedelta(days=3))
                    dt_ = st.time_input("마감 시각", value=time(23, 59))
                    p = st.selectbox("우선순위", [1, 2, 3], index=1)
                    if st.form_submit_button("추가") and n:
                        ok, msg = apply_op(state, "add_task", {"name": n, "duration_hours": h,
                                                               "deadline": f"{dd} {dt_:%H:%M}", "priority": p})
                        (st.success if ok else st.error)(msg)
                        st.session_state.last_log = replan(state)[3]
                        st.rerun()

        if view == "🧾 변경 로그":
            st.code(st.session_state.get("last_log", ""), language=None)
            st.markdown("**향후 계획**")
            st.text("\n".join(fmt_block(state, b) for b in state.plan if b.status == "planned") or "(없음)")

# ============================ 기본 ChatGPT ==================================
if svc == "💬 기본 ChatGPT":
    st.caption("같은 모델에 캘린더·작업·규칙을 텍스트로 주고, 계획을 직접 작성하게 하는 일반적인 ChatGPT 사용 방식입니다 (평가의 baseline과 같은 프롬프트).")
    bleft, bright = st.columns([5, 7], gap="large")
    with bleft:
        bbox = st.container(height=640)
        for m in st.session_state.get("base_messages", []):
            with bbox.chat_message(m["role"], avatar="💬" if m["role"] == "assistant" else None):
                st.markdown(md_safe(m["content"]))
        if bprompt := st.chat_input("기본 ChatGPT에게 말하기", key="base_input"):
            st.session_state.base_messages.append({"role": "user", "content": bprompt})
            with st.spinner("생각 중..."):
                try:
                    if st.session_state.get("baseline") is None:
                        snap = st.session_state.boot_state.clone()   # calendar + boot tasks only, never 일정 코치's changes
                        snap.now = state.now
                        start_baseline(snap, model)
                        st.session_state.base_messages.append({"role": "user", "content": bprompt})
                    resp = st.session_state.baseline.follow_up(fmt_dt(state.now), weekday_ko(state.now.date()), bprompt)
                    st.session_state.base_resp = resp
                    st.session_state.base_messages.append({"role": "assistant", "content": resp.get("explanation", "")})
                except Exception as e:  # noqa: BLE001
                    st.session_state.base_messages.append({"role": "assistant", "content": f"⚠️ 오류: {e}"})
            st.rerun()

    with bright:
        resp = st.session_state.get("base_resp") or {}
        bblocks, _ = parse_plan(resp.get("blocks", []))
        bview = seg("보기", ["📅 주간 계획", "🧾 계획 목록"], key="base_view")
        if bview == "📅 주간 계획":
            dfb = frame(state, [(n, s, e, "done" if e <= state.now else "") for n, s, e in bblocks], show_blocked=False)
            if not dfb.empty:
                chart(dfb, state, [n for n, _, _ in bblocks], key="chart_base")
            if resp.get("infeasible"):
                un = ", ".join(f"{u.get('task')} {u.get('hours')}시간" for u in resp.get("unscheduled", []) if isinstance(u, dict))
                st.warning("기본 ChatGPT 판단: 모두 배치할 수 없음" + (f" ({un})" if un else ""))
            elif resp:
                st.info("기본 ChatGPT 판단: 모두 배치 가능")
            st.caption("기본 ChatGPT가 출력한 블록을 그대로 그린 것입니다. 회색 = 캘린더 일정 · 빨간 선 = 현재 시각")
        if bview == "🧾 계획 목록":
            st.text("\n".join(f"{s:%m/%d}({weekday_ko(s.date())}) {s:%H:%M}-{e:%H:%M} {n}" for n, s, e in bblocks) or "(없음)")
