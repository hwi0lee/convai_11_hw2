"""Calendar sources: Google Calendar (read-only) and a relative-date mock for demos/evaluation."""
from __future__ import annotations

import json
import os
from datetime import datetime, time, timedelta
from typing import List, Tuple

from .models import Event, State, Task

GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]  # read-only by design


WEEKDAYS = "월화수목금토일"


def load_mock(path: str, now: datetime, with_tasks: bool = True) -> Tuple[List[Event], List[Task]]:
    """Mock calendar. Entries are defined by weekday ("weekday": "월") and mapped to the first
    such day on/after `now` (so classes stay on weekdays whatever the start date), or by a
    day offset ("day": 2) for backward compatibility."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    base = now.date()

    def offset(entry: dict, wd_key: str, day_key: str) -> int:
        if wd_key in entry:
            return (WEEKDAYS.index(entry[wd_key]) - base.weekday()) % 7
        return int(entry[day_key])

    def at(day: int, hm: str) -> datetime:
        if hm == "24:00":
            return datetime.combine(base + timedelta(days=day + 1), time(0))
        h, m = map(int, hm.split(":"))
        return datetime.combine(base + timedelta(days=day), time(h, m))

    events = [Event(at(offset(e, "weekday", "day"), e["start"]), at(offset(e, "weekday", "day"), e["end"]),
                    e.get("title", "일정")) for e in data["events"]]
    tasks = [Task(t["name"], int(t["hours"] * 60), at(offset(t, "deadline_weekday", "deadline_day"), t.get("deadline_time", "23:59")),
                  t.get("priority", 2), int(t.get("max_block_hours", 2) * 60)) for t in data["tasks"]] if with_tasks else []
    return events, tasks


def load_google(now: datetime, days: int = 14, credentials: str = "credentials.json",
                token: str = "token.json") -> List[Event]:
    """Fetch busy events from the primary Google Calendar. Requires an OAuth desktop-client
    `credentials.json` from Google Cloud Console (Calendar API enabled)."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(token):
        creds = Credentials.from_authorized_user_file(token, GOOGLE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials, GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token, "w") as f:
            f.write(creds.to_json())

    svc = build("calendar", "v3", credentials=creds)
    local_tz = datetime.now().astimezone().tzinfo
    t0 = datetime.combine(now.date(), time(0)).replace(tzinfo=local_tz)
    t1 = t0 + timedelta(days=days + 1)
    items = svc.events().list(calendarId="primary", timeMin=t0.isoformat(), timeMax=t1.isoformat(),
                              singleEvents=True, orderBy="startTime", maxResults=500).execute().get("items", [])
    out = []
    for it in items:
        if it.get("transparency") == "transparent":   # "free" events do not block time
            continue
        s, e = it["start"], it["end"]
        if "dateTime" in s:
            sd = datetime.fromisoformat(s["dateTime"].replace("Z", "+00:00")).astimezone(local_tz).replace(tzinfo=None)
            ed = datetime.fromisoformat(e["dateTime"].replace("Z", "+00:00")).astimezone(local_tz).replace(tzinfo=None)
        else:  # all-day event: treat as busy during waking hours
            sd = datetime.fromisoformat(s["date"]).replace(hour=7)
            ed = datetime.fromisoformat(e["date"]) - timedelta(days=1)
            ed = ed.replace(hour=23, minute=59)
        out.append(Event(sd, ed, it.get("summary", "일정")))
    return out


def initial_state(now: datetime, events: List[Event], tasks: List[Task]) -> State:
    return State(now=now, events=events, tasks=tasks)
