"""
Syncs data from Garmin Connect into the local SQLite database.
Handles session caching so repeated syncs don't re-login.
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
import requests

import database as db

logger = logging.getLogger(__name__)

# /tmp survives the process lifetime on both local and Streamlit Cloud.
# Falls back to data/ if /tmp isn't writable (shouldn't happen).
_TMP = Path("/tmp/garmin_tokens")
TOKEN_DIR = _TMP if _TMP.parent.exists() else Path(__file__).parent.parent / "data" / "garmin_tokens"


def _load_api() -> Optional[Garmin]:
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")

    if not email or not password:
        raise ValueError("GARMIN_EMAIL and GARMIN_PASSWORD must be set in .env")

    TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    try:
        api = Garmin(email, password)
        api.login(tokenstore=str(TOKEN_DIR))
        return api
    except (
        GarminConnectConnectionError,
        GarminConnectAuthenticationError,
        GarminConnectTooManyRequestsError,
        requests.exceptions.HTTPError,
    ) as e:
        logger.error("Garmin login failed: %s", e)
        return None


# ── Parsers ──────────────────────────────────────────────────────────────────

def _parse_sleep(raw: dict) -> Optional[dict]:
    cal = raw.get("calendarDate")
    if not cal:
        return None

    deep = raw.get("deepSleepSeconds", 0) or 0
    light = raw.get("lightSleepSeconds", 0) or 0
    rem = raw.get("remSleepSeconds", 0) or 0
    awake = raw.get("awakeSleepSeconds", 0) or 0
    total = deep + light + rem

    spo2 = raw.get("spo2SleepSummary") or {}

    return {
        "date": cal,
        "sleep_start_gmt": raw.get("sleepStartTimestampGMT"),
        "sleep_end_gmt": raw.get("sleepEndTimestampGMT"),
        "total_seconds": total,
        "deep_seconds": deep,
        "light_seconds": light,
        "rem_seconds": rem,
        "awake_seconds": awake,
        "avg_spo2": spo2.get("averageSPO2"),
        "lowest_spo2": spo2.get("lowestSPO2"),
        "avg_hr_during_sleep": spo2.get("averageHR"),
        "hrv_nightly_avg": raw.get("avgSleepStress"),
        "respiration_avg": raw.get("averageRespirationValue"),
        "confirmed_type": raw.get("sleepWindowConfirmationType"),
        "raw_json": json.dumps(raw),
    }


def _parse_daily_summary(raw: dict) -> Optional[dict]:
    cal = raw.get("calendarDate")
    if not cal:
        return None

    cal_str = cal if isinstance(cal, str) else cal.get("date", "")[:10] if isinstance(cal, dict) else str(cal)

    stress = {}
    all_day_stress = raw.get("allDayStress")
    if all_day_stress:
        aggs = all_day_stress.get("aggregatorList", [])
        for a in aggs:
            if a.get("type") == "AWAKE":
                stress = a
                break
        if not stress and aggs:
            stress = aggs[0]

    bb = raw.get("bodyBattery") or {}

    return {
        "date": cal_str[:10] if len(cal_str) >= 10 else cal_str,
        "total_steps": raw.get("totalSteps"),
        "step_goal": raw.get("dailyStepGoal"),
        "active_calories": raw.get("activeKilocalories"),
        "total_calories": raw.get("totalKilocalories"),
        "distance_meters": raw.get("totalDistanceMeters"),
        "active_seconds": raw.get("activeSeconds"),
        "highly_active_seconds": raw.get("highlyActiveSeconds"),
        "resting_hr": raw.get("restingHeartRate"),
        "min_hr": raw.get("minHeartRate"),
        "max_hr": raw.get("maxHeartRate"),
        "avg_stress": stress.get("averageStressLevel"),
        "max_stress": stress.get("maxStressLevel"),
        "stress_duration_seconds": stress.get("stressDuration"),
        "body_battery_charged": bb.get("charged"),
        "body_battery_drained": bb.get("drained"),
        "body_battery_highest": bb.get("highest"),
        "body_battery_lowest": bb.get("lowest"),
        "floors_ascended": raw.get("floorsAscendedInMeters"),
        "raw_json": json.dumps(raw, default=str),
    }


def _parse_activity(raw: dict) -> Optional[dict]:
    act_id = str(raw.get("activityId") or raw.get("summarizedActivitiesExport", {}).get("activityId", ""))
    if not act_id:
        return None

    ts = raw.get("startTimeLocal") or raw.get("startTimeGmt")
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts / 1000)
        date_str = dt.date().isoformat()
        start_str = dt.isoformat()
    elif isinstance(ts, str):
        date_str = ts[:10]
        start_str = ts
    else:
        date_str = None
        start_str = None

    duration_ms = raw.get("duration", 0) or 0
    duration_s = duration_ms / 1000 if duration_ms > 3600 else duration_ms

    return {
        "activity_id": act_id,
        "date": date_str,
        "name": raw.get("activityName") or raw.get("name"),
        "activity_type": raw.get("activityType", {}).get("typeKey") if isinstance(raw.get("activityType"), dict) else raw.get("activityType"),
        "sport_type": raw.get("sportType"),
        "start_time_local": start_str,
        "duration_seconds": duration_s,
        "distance_meters": raw.get("distance"),
        "avg_hr": raw.get("averageHR") or raw.get("avgHr"),
        "max_hr": raw.get("maxHR") or raw.get("maxHr"),
        "calories": raw.get("calories"),
        "avg_speed": raw.get("averageSpeed") or raw.get("avgSpeed"),
        "max_speed": raw.get("maxSpeed"),
        "training_effect": raw.get("trainingEffect"),
        "aerobic_effect": raw.get("aerobicTrainingEffect"),
        "anaerobic_effect": raw.get("anaerobicTrainingEffect"),
        "raw_json": json.dumps(raw, default=str),
    }


def _parse_fitness_age(raw: dict) -> Optional[dict]:
    as_of = raw.get("asOfDateGmt")
    if not as_of:
        return None
    date_str = as_of if isinstance(as_of, str) else as_of.get("date", "")[:12].strip()
    # "Jan 18, 2023 12:00:00 AM" → parse
    try:
        dt = datetime.strptime(date_str[:11].strip(), "%b %d, %Y")
        date_str = dt.date().isoformat()
    except ValueError:
        date_str = date_str[:10]

    return {
        "date": date_str,
        "chronological_age": raw.get("chronologicalAge"),
        "bio_age": raw.get("currentBioAge"),
        "vo2_max": raw.get("biometricVo2Max"),
        "rhr": raw.get("rhr"),
        "bmi": raw.get("bmi"),
        "total_vigorous_days": raw.get("totalVigorousDays"),
        "vigorous_intensity_minutes": raw.get("totalVigorousIMs"),
    }


# ── Live sync from Garmin Connect API ────────────────────────────────────────

def sync_range(start_date: date, end_date: date, progress_cb=None) -> dict:
    """Pull data for a date range from Garmin Connect and store in SQLite."""
    api = _load_api()
    if not api:
        return {"error": "Could not connect to Garmin Connect. Check credentials."}

    counts = {"sleep": 0, "summaries": 0, "activities": 0}
    current = start_date

    while current <= end_date:
        d = current.isoformat()
        if progress_cb:
            progress_cb(f"Syncing {d}…")

        # Sleep
        try:
            raw_sleep = api.get_sleep_data(d)
            if isinstance(raw_sleep, dict):
                daily = raw_sleep.get("dailySleepDTO") or raw_sleep
                parsed = _parse_sleep(daily)
                if parsed:
                    db.upsert_sleep([parsed])
                    counts["sleep"] += 1
            elif isinstance(raw_sleep, list):
                parsed_list = [p for r in raw_sleep if (p := _parse_sleep(r))]
                if parsed_list:
                    db.upsert_sleep(parsed_list)
                    counts["sleep"] += len(parsed_list)
        except Exception as e:
            logger.warning("Sleep fetch failed for %s: %s", d, e)

        # Daily summary
        try:
            raw_summary = api.get_stats(d)
            if raw_summary:
                parsed = _parse_daily_summary(raw_summary)
                if parsed:
                    db.upsert_daily_summaries([parsed])
                    counts["summaries"] += 1
        except Exception as e:
            logger.warning("Summary fetch failed for %s: %s", d, e)

        current += timedelta(days=1)

    # Activities (bulk fetch for the range)
    try:
        raw_activities = api.get_activities_by_date(
            start_date.isoformat(), end_date.isoformat()
        )
        parsed_acts = [p for r in (raw_activities or []) if (p := _parse_activity(r))]
        if parsed_acts:
            db.upsert_activities(parsed_acts)
            counts["activities"] = len(parsed_acts)
    except Exception as e:
        logger.warning("Activities fetch failed: %s", e)

    return counts


def sync_recent(days: int = 30, progress_cb=None) -> dict:
    today = date.today()
    start = today - timedelta(days=days)
    return sync_range(start, today, progress_cb)


def get_connection_status() -> dict:
    try:
        api = _load_api()
        if api:
            name = api.get_full_name()
            return {"connected": True, "name": name}
        return {"connected": False, "error": "Login failed"}
    except Exception as e:
        return {"connected": False, "error": str(e)}
