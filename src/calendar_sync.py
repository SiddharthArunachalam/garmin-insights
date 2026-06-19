"""
Google Calendar integration.
First run: python src/calendar_sync.py --auth   (opens browser for OAuth)
Subsequent syncs happen automatically using the saved token.
"""

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import database as db

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
CREDENTIALS_FILE = Path(__file__).parent.parent / "data" / "google_credentials.json"
TOKEN_FILE = Path(__file__).parent.parent / "data" / "google_token.json"

# Keywords used to tag event types
_WORKOUT_KW = re.compile(
    r"gym|workout|run|yoga|cycling|swim|lift|crossfit|hiit|pilates|train|exercise|cardio|walk",
    re.I,
)
_TRAVEL_KW = re.compile(
    r"flight|travel|trip|hotel|airport|depart|arrive|layover|vacation|holiday",
    re.I,
)
_LATE_NIGHT_KW = re.compile(
    r"dinner|party|concert|show|gala|event|drinks|bar|night",
    re.I,
)


def _classify_event(title: str, description: str = "") -> str:
    text = f"{title} {description}"
    if _WORKOUT_KW.search(text):
        return "workout"
    if _TRAVEL_KW.search(text):
        return "travel"
    if _LATE_NIGHT_KW.search(text):
        return "social"
    return "other"


def _get_credentials() -> Credentials:
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Google credentials file not found at {CREDENTIALS_FILE}.\n"
                    "Download it from Google Cloud Console > APIs & Services > Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return creds


def _parse_event(event: dict, calendar_name: str) -> dict:
    title = event.get("summary", "")
    description = event.get("description", "") or ""
    event_id = event.get("id", "")

    start = event.get("start", {})
    end = event.get("end", {})

    start_dt = start.get("dateTime") or start.get("date", "")
    end_dt = end.get("dateTime") or end.get("date", "")
    date_str = start_dt[:10] if start_dt else ""

    return {
        "event_id": event_id,
        "date": date_str,
        "title": title,
        "start_datetime": start_dt,
        "end_datetime": end_dt,
        "event_type": _classify_event(title, description),
        "calendar_name": calendar_name,
        "description": description[:500],
    }


def sync_range(start_date: date, end_date: date) -> dict:
    """Fetch Google Calendar events for a date range and store in SQLite."""
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)

    time_min = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
    time_max = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).isoformat() + "Z"

    calendars = service.calendarList().list().execute().get("items", [])
    all_events = []

    for cal in calendars:
        cal_id = cal["id"]
        cal_name = cal.get("summary", cal_id)
        page_token = None

        while True:
            result = service.events().list(
                calendarId=cal_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
                maxResults=500,
                pageToken=page_token,
            ).execute()

            for event in result.get("items", []):
                all_events.append(_parse_event(event, cal_name))

            page_token = result.get("nextPageToken")
            if not page_token:
                break

    if all_events:
        db.upsert_calendar_events(all_events)

    return {"events_synced": len(all_events)}


def sync_recent(days: int = 90) -> dict:
    today = date.today()
    return sync_range(today - timedelta(days=days), today + timedelta(days=30))


def is_authenticated() -> bool:
    if not TOKEN_FILE.exists():
        return False
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if creds and creds.valid:
            return True
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            return True
    except Exception:
        pass
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", action="store_true", help="Run OAuth flow")
    parser.add_argument("--sync", action="store_true", help="Sync recent events")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))

    if args.auth:
        print("Opening browser for Google Calendar authorization…")
        _get_credentials()
        print("Authorization complete. Token saved.")
    elif args.sync:
        result = sync_recent()
        print(f"Synced {result['events_synced']} calendar events.")
    else:
        parser.print_help()
