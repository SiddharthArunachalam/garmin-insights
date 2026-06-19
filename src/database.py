"""
Database layer — SQLite locally, PostgreSQL (Supabase) in production.
Set DATABASE_URL env var to switch to PostgreSQL.
"""

import os
import json
from pathlib import Path
from typing import Optional

_DB_URL  = os.getenv("DATABASE_URL")
_BACKEND = "pg" if _DB_URL else "sqlite"

if _BACKEND == "pg":
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3

DB_PATH = Path(__file__).parent.parent / "data" / "garmin_insights.db"


# ── Connection helpers ────────────────────────────────────────────────────────

def get_conn():
    if _BACKEND == "pg":
        return psycopg2.connect(_DB_URL)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _fetchall(sql: str, params: tuple = ()):
    with get_conn() as conn:
        if _BACKEND == "pg":
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        return conn.execute(sql, params).fetchall()


def _fetchone(sql: str, params: tuple = ()):
    with get_conn() as conn:
        if _BACKEND == "pg":
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return cur.fetchone()
        return conn.execute(sql, params).fetchone()


def _execmany(sql: str, rows: list[dict]):
    with get_conn() as conn:
        if _BACKEND == "pg":
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, rows)
        else:
            conn.executemany(sql, rows)


# Placeholder token: ? for SQLite, %s for PostgreSQL
def _p() -> str:
    return "%s" if _BACKEND == "pg" else "?"


def _upsert(table: str, pk: str | list, cols: list[str]) -> str:
    """Build an UPSERT statement for either backend."""
    col_list = ", ".join(cols)
    pks = [pk] if isinstance(pk, str) else pk
    if _BACKEND == "pg":
        vals      = ", ".join(f"%({c})s" for c in cols)
        non_pk    = [c for c in cols if c not in pks]
        updates   = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk)
        conflict  = ", ".join(pks)
        return (f"INSERT INTO {table} ({col_list}) VALUES ({vals}) "
                f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}")
    else:
        vals = ", ".join(f":{c}" for c in cols)
        return f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({vals})"


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    _now = "NOW()" if _BACKEND == "pg" else "datetime('now')"
    stmts = [
        """CREATE TABLE IF NOT EXISTS sleep_records (
            date TEXT PRIMARY KEY,
            sleep_start_gmt TEXT,
            sleep_end_gmt TEXT,
            total_seconds INTEGER,
            deep_seconds INTEGER,
            light_seconds INTEGER,
            rem_seconds INTEGER,
            awake_seconds INTEGER,
            avg_spo2 REAL,
            lowest_spo2 REAL,
            avg_hr_during_sleep REAL,
            hrv_nightly_avg REAL,
            respiration_avg REAL,
            confirmed_type TEXT,
            raw_json TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS daily_summaries (
            date TEXT PRIMARY KEY,
            total_steps INTEGER,
            step_goal INTEGER,
            active_calories INTEGER,
            total_calories INTEGER,
            distance_meters REAL,
            active_seconds INTEGER,
            highly_active_seconds INTEGER,
            resting_hr INTEGER,
            min_hr INTEGER,
            max_hr INTEGER,
            avg_stress INTEGER,
            max_stress INTEGER,
            stress_duration_seconds INTEGER,
            body_battery_charged INTEGER,
            body_battery_drained INTEGER,
            body_battery_highest INTEGER,
            body_battery_lowest INTEGER,
            floors_ascended REAL,
            raw_json TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS activities (
            activity_id TEXT PRIMARY KEY,
            date TEXT,
            name TEXT,
            activity_type TEXT,
            sport_type TEXT,
            start_time_local TEXT,
            duration_seconds REAL,
            distance_meters REAL,
            avg_hr INTEGER,
            max_hr INTEGER,
            calories REAL,
            avg_speed REAL,
            max_speed REAL,
            training_effect REAL,
            aerobic_effect REAL,
            anaerobic_effect REAL,
            raw_json TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS fitness_age (
            date TEXT PRIMARY KEY,
            chronological_age INTEGER,
            bio_age REAL,
            vo2_max REAL,
            rhr INTEGER,
            bmi REAL,
            total_vigorous_days INTEGER,
            vigorous_intensity_minutes INTEGER
        )""",
        f"""CREATE TABLE IF NOT EXISTS sleep_feedback (
            date TEXT PRIMARY KEY,
            subjective_quality INTEGER,
            energy_on_waking INTEGER,
            notes TEXT,
            alcohol_drinks INTEGER,
            late_meal INTEGER,
            stress_level INTEGER,
            caffeine_after_2pm INTEGER,
            created_at TEXT DEFAULT ({_now})
        )""",
        """CREATE TABLE IF NOT EXISTS calendar_events (
            event_id TEXT PRIMARY KEY,
            date TEXT,
            title TEXT,
            start_datetime TEXT,
            end_datetime TEXT,
            event_type TEXT,
            calendar_name TEXT,
            description TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_activities_date ON activities(date)",
        "CREATE INDEX IF NOT EXISTS idx_calendar_date ON calendar_events(date)",
    ]

    with get_conn() as conn:
        if _BACKEND == "pg":
            with conn.cursor() as cur:
                for stmt in stmts:
                    cur.execute(stmt)
        else:
            for stmt in stmts:
                conn.execute(stmt)


# ── Sleep ─────────────────────────────────────────────────────────────────────

_SLEEP_COLS = [
    "date", "sleep_start_gmt", "sleep_end_gmt", "total_seconds", "deep_seconds",
    "light_seconds", "rem_seconds", "awake_seconds", "avg_spo2", "lowest_spo2",
    "avg_hr_during_sleep", "hrv_nightly_avg", "respiration_avg", "confirmed_type", "raw_json",
]

def upsert_sleep(records: list[dict]):
    _execmany(_upsert("sleep_records", "date", _SLEEP_COLS), records)


def get_sleep(start: str, end: str):
    p = _p()
    return _fetchall(
        f"SELECT * FROM sleep_records WHERE date BETWEEN {p} AND {p} ORDER BY date",
        (start, end),
    )


def get_sleep_for_date(d: str):
    return _fetchone(f"SELECT * FROM sleep_records WHERE date = {_p()}", (d,))


# ── Daily summaries ───────────────────────────────────────────────────────────

_DAILY_COLS = [
    "date", "total_steps", "step_goal", "active_calories", "total_calories",
    "distance_meters", "active_seconds", "highly_active_seconds", "resting_hr",
    "min_hr", "max_hr", "avg_stress", "max_stress", "stress_duration_seconds",
    "body_battery_charged", "body_battery_drained", "body_battery_highest",
    "body_battery_lowest", "floors_ascended", "raw_json",
]

def upsert_daily_summaries(records: list[dict]):
    _execmany(_upsert("daily_summaries", "date", _DAILY_COLS), records)


def get_daily_summaries(start: str, end: str):
    p = _p()
    return _fetchall(
        f"SELECT * FROM daily_summaries WHERE date BETWEEN {p} AND {p} ORDER BY date",
        (start, end),
    )


# ── Activities ────────────────────────────────────────────────────────────────

_ACT_COLS = [
    "activity_id", "date", "name", "activity_type", "sport_type", "start_time_local",
    "duration_seconds", "distance_meters", "avg_hr", "max_hr", "calories",
    "avg_speed", "max_speed", "training_effect", "aerobic_effect", "anaerobic_effect", "raw_json",
]

def upsert_activities(records: list[dict]):
    _execmany(_upsert("activities", "activity_id", _ACT_COLS), records)


def get_activities(start: str, end: str):
    p = _p()
    return _fetchall(
        f"SELECT * FROM activities WHERE date BETWEEN {p} AND {p} ORDER BY date DESC",
        (start, end),
    )


# ── Fitness age ───────────────────────────────────────────────────────────────

_FITNESS_COLS = [
    "date", "chronological_age", "bio_age", "vo2_max", "rhr", "bmi",
    "total_vigorous_days", "vigorous_intensity_minutes",
]

def upsert_fitness_age(records: list[dict]):
    _execmany(_upsert("fitness_age", "date", _FITNESS_COLS), records)


def get_fitness_age(start: str, end: str):
    p = _p()
    return _fetchall(
        f"SELECT * FROM fitness_age WHERE date BETWEEN {p} AND {p} ORDER BY date",
        (start, end),
    )


# ── Sleep feedback ────────────────────────────────────────────────────────────

_FEEDBACK_COLS = [
    "date", "subjective_quality", "energy_on_waking", "notes",
    "alcohol_drinks", "late_meal", "stress_level", "caffeine_after_2pm",
]

def upsert_sleep_feedback(record: dict):
    _execmany(_upsert("sleep_feedback", "date", _FEEDBACK_COLS), [record])


def get_sleep_feedback(start: str, end: str):
    p = _p()
    return _fetchall(
        f"SELECT * FROM sleep_feedback WHERE date BETWEEN {p} AND {p} ORDER BY date",
        (start, end),
    )


def get_feedback_for_date(d: str):
    return _fetchone(f"SELECT * FROM sleep_feedback WHERE date = {_p()}", (d,))


# ── Calendar events ───────────────────────────────────────────────────────────

_CAL_COLS = [
    "event_id", "date", "title", "start_datetime", "end_datetime",
    "event_type", "calendar_name", "description",
]

def upsert_calendar_events(records: list[dict]):
    _execmany(_upsert("calendar_events", "event_id", _CAL_COLS), records)


def get_calendar_events(start: str, end: str):
    p = _p()
    return _fetchall(
        f"SELECT * FROM calendar_events WHERE date BETWEEN {p} AND {p} ORDER BY date, start_datetime",
        (start, end),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_date_range_with_data() -> tuple[Optional[str], Optional[str]]:
    row = _fetchone("SELECT MIN(date) AS mn, MAX(date) AS mx FROM sleep_records")
    if row:
        return row["mn"], row["mx"]
    return None, None


def has_feedback_for_date(d: str) -> bool:
    row = _fetchone(f"SELECT 1 FROM sleep_feedback WHERE date = {_p()}", (d,))
    return row is not None
