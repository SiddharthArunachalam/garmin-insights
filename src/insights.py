"""
Gemini-powered insights engine.
Uses google-genai (gemini-2.0-flash) — free tier, no credit card needed.
Get a key at https://aistudio.google.com
"""

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types
import database as db

_client = None
MODEL = "gemini-2.5-flash"


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env — get one free at https://aistudio.google.com")
        _client = genai.Client(api_key=api_key)
    return _client


# ── Data serialisation helpers ────────────────────────────────────────────────

def _fmt_duration(seconds) -> str:
    if not seconds:
        return "0h 0m"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h {m}m"


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


def _build_sleep_context(start: str, end: str) -> str:
    sleep = _rows_to_dicts(db.get_sleep(start, end))
    feedback = _rows_to_dicts(db.get_sleep_feedback(start, end))
    fb_by_date = {f["date"]: f for f in feedback}

    if not sleep:
        return "No sleep data available for this period."

    lines = ["DATE | TOTAL | DEEP | REM | LIGHT | AWAKE | SpO2avg | SpO2low | HR | Subj | Energy | Alcohol | Stress"]
    for s in sleep:
        fb = fb_by_date.get(s["date"], {})
        lines.append(
            f"{s['date']} | "
            f"{_fmt_duration(s['total_seconds'])} | "
            f"{_fmt_duration(s['deep_seconds'])} | "
            f"{_fmt_duration(s['rem_seconds'])} | "
            f"{_fmt_duration(s['light_seconds'])} | "
            f"{_fmt_duration(s['awake_seconds'])} | "
            f"{s['avg_spo2'] or 'N/A'} | "
            f"{s['lowest_spo2'] or 'N/A'} | "
            f"{s['avg_hr_during_sleep'] or 'N/A'} | "
            f"{fb.get('subjective_quality', '?')}/10 | "
            f"{fb.get('energy_on_waking', '?')}/10 | "
            f"{fb.get('alcohol_drinks', '?')} drinks | "
            f"{fb.get('stress_level', '?')}/10"
        )
    return "\n".join(lines)


def _build_activity_context(start: str, end: str) -> str:
    acts = _rows_to_dicts(db.get_activities(start, end))
    if not acts:
        return "No workout activity data available."

    lines = ["DATE | NAME | TYPE | DURATION | DIST(km) | AvgHR | MaxHR | CALORIES"]
    for a in acts:
        dist_km = round(a["distance_meters"] / 1000, 2) if a["distance_meters"] else "N/A"
        lines.append(
            f"{a['date']} | {a['name'] or 'Unknown'} | {a['activity_type'] or ''} | "
            f"{_fmt_duration(a['duration_seconds'])} | "
            f"{dist_km} | {a['avg_hr'] or 'N/A'} | {a['max_hr'] or 'N/A'} | {int(a['calories'] or 0)}"
        )
    return "\n".join(lines)


def _build_daily_context(start: str, end: str) -> str:
    rows = _rows_to_dicts(db.get_daily_summaries(start, end))
    if not rows:
        return "No daily summary data available."

    lines = ["DATE | STEPS | RHR | AVG_STRESS | MAX_STRESS | BB_HIGH | BB_LOW | CALORIES"]
    for r in rows:
        lines.append(
            f"{r['date']} | {r['total_steps'] or 'N/A'} | {r['resting_hr'] or 'N/A'} | "
            f"{r['avg_stress'] or 'N/A'} | {r['max_stress'] or 'N/A'} | "
            f"{r['body_battery_highest'] or 'N/A'} | {r['body_battery_lowest'] or 'N/A'} | "
            f"{int(r['total_calories'] or 0)}"
        )
    return "\n".join(lines)


def _build_calendar_context(start: str, end: str) -> str:
    events = _rows_to_dicts(db.get_calendar_events(start, end))
    if not events:
        return "No calendar events available."

    lines = ["DATE | TYPE | TITLE | START | END"]
    for e in events:
        lines.append(
            f"{e['date']} | {e['event_type']} | {e['title']} | "
            f"{e['start_datetime'] or ''} | {e['end_datetime'] or ''}"
        )
    return "\n".join(lines)


def _build_fitness_context(start: str, end: str) -> str:
    rows = _rows_to_dicts(db.get_fitness_age(start, end))
    if not rows:
        return "No fitness age / VO2 max data available."

    lines = ["DATE | VO2_MAX | BIO_AGE | RHR | BMI | VIGOROUS_DAYS"]
    for r in rows:
        lines.append(
            f"{r['date']} | {round(r['vo2_max'], 1) if r['vo2_max'] else 'N/A'} | "
            f"{round(r['bio_age'], 1) if r['bio_age'] else 'N/A'} | "
            f"{r['rhr'] or 'N/A'} | {round(r['bmi'], 1) if r['bmi'] else 'N/A'} | "
            f"{r['total_vigorous_days'] or 'N/A'}"
        )
    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a personal health coach and data analyst with expertise in sleep science,
exercise physiology, and cardiovascular health. You have access to the user's Garmin Vivo Active 5
health data including sleep metrics, daily activity summaries, workout logs, fitness age/VO2 max
trends, and Google Calendar events.

Your role is to:
1. Identify patterns and correlations in the data
2. Explain what the metrics mean in plain language
3. Surface actionable insights (e.g. "Your deep sleep drops significantly after alcohol or late meals")
4. Track cardio fitness trends (VO2 max, resting HR, HRV)
5. Correlate calendar events (travel, late nights, heavy workouts) with recovery and sleep quality
6. Ask clarifying questions when user feedback would improve the analysis

Be specific — reference actual dates and numbers from the data. Avoid generic health advice.
When you spot a meaningful correlation, explain the likely mechanism in 1-2 sentences.

Format insights as:
- **Finding**: what the data shows
- **Why it matters**: brief explanation
- **Suggestion**: one concrete action"""


def build_context_block(days: int = 30) -> str:
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days)).isoformat()

    return f"""## Health Data Context ({start} to {end})

### Sleep Data
{_build_sleep_context(start, end)}

### Daily Activity Summaries
{_build_daily_context(start, end)}

### Workouts
{_build_activity_context(start, end)}

### Fitness Age & VO2 Max
{_build_fitness_context(start, end)}

### Calendar Events
{_build_calendar_context(start, end)}
"""


# ── Chat interface ────────────────────────────────────────────────────────────

def ask(
    question: str,
    history: list[dict],
    days_of_context: int = 30,
) -> Generator[str, None, None]:
    """Stream a response from Gemini given a question and chat history."""
    client = _get_client()
    context = build_context_block(days_of_context)

    contents = []
    if history:
        contents.append({"role": "user", "parts": [{"text": context}]})
        contents.append({"role": "model", "parts": [{"text": "I've reviewed your health data. How can I help you?"}]})
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": question}]})
    else:
        contents.append({"role": "user", "parts": [{"text": f"{context}\n\n---\n\n{question}"}]})

    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
        ),
    ):
        if chunk.text:
            yield chunk.text


def generate_sleep_insights(days: int = 14) -> Generator[str, None, None]:
    """Generate a proactive sleep analysis report."""
    model = _get_model()
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days)).isoformat()

    context = f"""## Health Data — Last {days} Days ({start} to {end})

### Sleep Data
{_build_sleep_context(start, end)}

### Daily Summaries (Stress, Steps, Body Battery)
{_build_daily_context(start, end)}

### Workouts
{_build_activity_context(start, end)}

### Calendar Events
{_build_calendar_context(start, end)}
"""

    prompt = f"""{context}

---

Analyse this person's sleep over the past {days} days.

Provide:
1. **Sleep Quality Summary** — overall trends, average totals, how deep/REM compare to healthy benchmarks (deep ~20%, REM ~25% of total)
2. **Top Factors Affecting Sleep** — find correlations with stress, workouts, calendar events, alcohol, body battery
3. **Best vs Worst Sleep Nights** — what was different on those nights?
4. **Cardio & Recovery Link** — how does sleep quality affect next-day resting HR and body battery?
5. **3 Actionable Recommendations** — specific to this person's data patterns

Be specific with dates and numbers."""

    client = _get_client()
    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
        ),
    ):
        if chunk.text:
            yield chunk.text


def generate_cardio_insights(days: int = 30) -> Generator[str, None, None]:
    """Generate a cardio fitness trend analysis."""
    model = _get_model()
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days)).isoformat()

    context = f"""## Health Data — Last {days} Days

### Workouts
{_build_activity_context(start, end)}

### Daily Summaries (Resting HR, Stress, Steps)
{_build_daily_context(start, end)}

### Fitness Age & VO2 Max
{_build_fitness_context(start, end)}

### Sleep Data
{_build_sleep_context(start, end)}
"""

    prompt = f"""{context}

---

Analyse this person's cardio fitness and workout trends over the past {days} days.

Provide:
1. **Cardio Fitness Trend** — VO2 max changes, resting HR trend, what they signal
2. **Workout Load Analysis** — frequency, intensity, variety; signs of overtraining or undertraining
3. **Recovery Quality** — how well is the person recovering between sessions? (body battery, RHR the next day)
4. **Sleep–Performance Link** — does sleep quality affect workout HR or performance?
5. **3 Recommendations** — to improve cardio fitness based on actual patterns

Reference specific workouts and dates."""

    client = _get_client()
    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
        ),
    ):
        if chunk.text:
            yield chunk.text
