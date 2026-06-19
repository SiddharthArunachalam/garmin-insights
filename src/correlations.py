"""
Statistical correlations engine.
Finds concrete patterns across sleep, cardio, activity, recovery, and lifestyle.
All maths is pure pandas — no AI API required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import database as db

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    category: str
    title: str
    detail: str
    direction: str   # "positive" | "negative" | "neutral"
    magnitude: float = 0.0


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_combined(days: int = 90) -> pd.DataFrame:
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days)).isoformat()

    sleep_rows    = [dict(r) for r in db.get_sleep(start, end)]
    daily_rows    = [dict(r) for r in db.get_daily_summaries(start, end)]
    feedback_rows = [dict(r) for r in db.get_sleep_feedback(start, end)]
    activity_rows = [dict(r) for r in db.get_activities(start, end)]
    calendar_rows = [dict(r) for r in db.get_calendar_events(start, end)]

    sleep_df    = pd.DataFrame(sleep_rows)    if sleep_rows    else pd.DataFrame()
    daily_df    = pd.DataFrame(daily_rows)    if daily_rows    else pd.DataFrame()
    feedback_df = pd.DataFrame(feedback_rows) if feedback_rows else pd.DataFrame()
    act_df      = pd.DataFrame(activity_rows) if activity_rows else pd.DataFrame()
    cal_df      = pd.DataFrame(calendar_rows) if calendar_rows else pd.DataFrame()

    if sleep_df.empty and daily_df.empty:
        return pd.DataFrame()

    # ── Activity aggregates per day ───────────────────────────────────────────
    if not act_df.empty and "date" in act_df.columns:
        act_df["date"] = act_df["date"].astype(str).str[:10]
        act_agg = act_df.groupby("date").agg(
            had_workout          = ("activity_id",    "count"),
            workout_avg_hr       = ("avg_hr",         "mean"),
            workout_max_hr       = ("max_hr",         "mean"),
            workout_duration_min = ("duration_seconds", lambda x: x.sum() / 60),
            workout_calories     = ("calories",       "sum"),
            workout_aerobic_eff  = ("aerobic_effect", "mean"),
        ).reset_index()
        act_agg["had_workout"] = act_agg["had_workout"] > 0
        # Dominant workout type per day
        type_mode = act_df.groupby("date")["activity_type"].agg(
            lambda x: x.mode().iloc[0] if len(x) > 0 else None
        ).reset_index().rename(columns={"activity_type": "workout_type"})
        act_agg = act_agg.merge(type_mode, on="date", how="left")
    else:
        act_agg = pd.DataFrame(columns=[
            "date", "had_workout", "workout_avg_hr", "workout_max_hr",
            "workout_duration_min", "workout_calories", "workout_aerobic_eff", "workout_type",
        ])

    # ── Calendar aggregates per day ───────────────────────────────────────────
    if not cal_df.empty and "date" in cal_df.columns:
        cal_df["date"] = cal_df["date"].astype(str).str[:10]
        cal_agg = cal_df.groupby("date").agg(
            had_travel  = ("event_type", lambda x: (x == "travel").any()),
            had_social  = ("event_type", lambda x: (x == "social").any()),
        ).reset_index()
    else:
        cal_agg = pd.DataFrame(columns=["date", "had_travel", "had_social"])

    # ── Sleep processing ──────────────────────────────────────────────────────
    if not sleep_df.empty:
        sleep_df["date"]       = sleep_df["date"].astype(str).str[:10]
        sleep_df["total_hours"]= sleep_df["total_seconds"] / 3600
        total                  = sleep_df["total_seconds"].replace(0, float("nan"))
        sleep_df["deep_pct"]   = (sleep_df["deep_seconds"]  / total * 100)
        sleep_df["rem_pct"]    = (sleep_df["rem_seconds"]   / total * 100)
        sleep_df["light_pct"]  = (sleep_df["light_seconds"] / total * 100)
        sleep_df["sleep_date"] = pd.to_datetime(sleep_df["date"])
        sleep_df["prev_date"]  = (sleep_df["sleep_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        sleep_df["next_date"]  = (sleep_df["sleep_date"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")

    if not daily_df.empty:
        daily_df["date"] = daily_df["date"].astype(str).str[:10]
        daily_df["bb_recovery"] = daily_df["body_battery_highest"] - daily_df["body_battery_lowest"]

    if not feedback_df.empty:
        feedback_df["date"]             = feedback_df["date"].astype(str).str[:10]
        feedback_df["had_alcohol"]      = feedback_df["alcohol_drinks"] > 0
        feedback_df["late_meal"]        = feedback_df["late_meal"].astype(bool)
        feedback_df["caffeine_after_2pm"] = feedback_df["caffeine_after_2pm"].astype(bool)

    # ── Primary join: sleep + same-day daily stats ────────────────────────────
    if not sleep_df.empty and not daily_df.empty:
        df = sleep_df.merge(daily_df, on="date", how="left", suffixes=("", "_daily"))
    elif not sleep_df.empty:
        df = sleep_df.copy()
    else:
        df = daily_df.copy()

    # Sleep feedback
    if not feedback_df.empty and "date" in df.columns:
        df = df.merge(
            feedback_df[["date", "subjective_quality", "energy_on_waking",
                          "alcohol_drinks", "had_alcohol", "late_meal",
                          "caffeine_after_2pm", "stress_level"]],
            on="date", how="left",
        )

    # Previous-day workout (workouts before that night's sleep)
    if not act_agg.empty and "prev_date" in df.columns:
        df = df.merge(act_agg.rename(columns={"date": "prev_date"}), on="prev_date", how="left")

    # Next-day daily stats (how sleep quality affects tomorrow)
    if not daily_df.empty and "next_date" in df.columns:
        next_cols = {c: f"next_{c}" for c in daily_df.columns if c != "date"}
        next_daily = daily_df.rename(columns=next_cols).rename(columns={"date": "next_date"})
        df = df.merge(next_daily, on="next_date", how="left")

    # Calendar flags
    if not cal_agg.empty and "date" in df.columns:
        df = df.merge(cal_agg, on="date", how="left")

    # Fill boolean flags — NaN after left join = no event that day
    for col in ["had_workout", "had_travel", "had_social"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df


# ── Shared helpers ────────────────────────────────────────────────────────────

def _mean_diff(df: pd.DataFrame, group_col: str, value_col: str) -> Optional[tuple]:
    """Return (mean_true, mean_false, n_true, n_false) or None."""
    if group_col not in df.columns or value_col not in df.columns:
        return None
    sub = df[[group_col, value_col]].dropna()
    if len(sub) < 5:
        return None
    t = sub[sub[group_col] == True][value_col]
    f = sub[sub[group_col] == False][value_col]
    if len(t) < 2 or len(f) < 2:
        return None
    return float(t.mean()), float(f.mean()), len(t), len(f)


def _bucket_diff(series: pd.Series, metric: pd.Series,
                 lo_thresh, hi_thresh) -> Optional[tuple]:
    """Compare metric for low vs high values of series."""
    df2 = pd.DataFrame({"x": series, "y": metric}).dropna()
    if len(df2) < 6:
        return None
    lo = df2[df2["x"] <= lo_thresh]["y"]
    hi = df2[df2["x"] >= hi_thresh]["y"]
    if len(lo) < 2 or len(hi) < 2:
        return None
    return float(lo.mean()), float(hi.mean()), len(lo), len(hi)


# ── Finders ───────────────────────────────────────────────────────────────────

def _find_workout_sleep(df: pd.DataFrame) -> list[Finding]:
    findings = []
    for metric, label, scale, unit in [
        ("total_hours", "total sleep", 60, "min"),
        ("deep_pct",    "deep sleep %", 1, "pp"),
        ("rem_pct",     "REM %",        1, "pp"),
    ]:
        r = _mean_diff(df, "had_workout", metric)
        if not r:
            continue
        with_wk, without_wk, n_with, n_without = r
        diff = (with_wk - without_wk) * scale
        if abs(diff) < (8 if unit == "min" else 1.5):
            continue
        better = diff > 0
        findings.append(Finding(
            category="Sleep vs Workouts",
            title=f"Workout days → {'more' if better else 'less'} {label}",
            detail=(
                f"Nights after a workout (n={n_with}): {with_wk:.1f}{'' if unit == 'pp' else 'h' if metric == 'total_hours' else '%'} "
                f"vs {without_wk:.1f} on rest days (n={n_without}), diff {diff:+.0f}{unit}."
            ),
            direction="positive" if better else "negative",
            magnitude=abs(diff),
        ))

    # Workout intensity vs sleep
    if "workout_avg_hr" in df.columns and "total_hours" in df.columns:
        sub = df[["workout_avg_hr", "total_hours", "deep_pct"]].dropna()
        if len(sub) >= 6:
            hi_hr = sub[sub["workout_avg_hr"] >= sub["workout_avg_hr"].quantile(0.67)]
            lo_hr = sub[sub["workout_avg_hr"] <= sub["workout_avg_hr"].quantile(0.33)]
            if len(hi_hr) >= 3 and len(lo_hr) >= 3:
                diff_min = (hi_hr["total_hours"].mean() - lo_hr["total_hours"].mean()) * 60
                if abs(diff_min) >= 8:
                    findings.append(Finding(
                        category="Sleep vs Workouts",
                        title=f"High-intensity workouts → {'more' if diff_min > 0 else 'less'} sleep",
                        detail=(
                            f"After hard workouts (avg HR ≥{hi_hr['workout_avg_hr'].mean():.0f} bpm, n={len(hi_hr)}): "
                            f"{hi_hr['total_hours'].mean():.1f}h sleep vs "
                            f"{lo_hr['total_hours'].mean():.1f}h after easier sessions (n={len(lo_hr)}), "
                            f"diff {diff_min:+.0f} min."
                        ),
                        direction="positive" if diff_min > 0 else "negative",
                        magnitude=abs(diff_min),
                    ))
                diff_deep = hi_hr["deep_pct"].mean() - lo_hr["deep_pct"].mean()
                if abs(diff_deep) >= 1.5:
                    findings.append(Finding(
                        category="Sleep vs Workouts",
                        title=f"High-intensity workouts → {'more' if diff_deep > 0 else 'less'} deep sleep",
                        detail=(
                            f"Deep sleep after hard workouts: {hi_hr['deep_pct'].mean():.1f}% "
                            f"vs {lo_hr['deep_pct'].mean():.1f}% after easier ones ({diff_deep:+.1f} pp)."
                        ),
                        direction="positive" if diff_deep > 0 else "negative",
                        magnitude=abs(diff_deep),
                    ))

    # Workout duration vs sleep
    if "workout_duration_min" in df.columns and "total_hours" in df.columns:
        sub = df[["workout_duration_min", "total_hours"]].dropna()
        if len(sub) >= 6:
            r = _bucket_diff(sub["workout_duration_min"], sub["total_hours"], 30, 60)
            if r:
                short_sleep, long_sleep, n_short, n_long = r
                diff_min = (long_sleep - short_sleep) * 60
                if abs(diff_min) >= 8:
                    findings.append(Finding(
                        category="Sleep vs Workouts",
                        title=f"Longer workouts → {'more' if diff_min > 0 else 'less'} sleep",
                        detail=(
                            f"After workouts ≥60 min (n={n_long}): {long_sleep:.1f}h sleep "
                            f"vs {short_sleep:.1f}h after ≤30 min sessions (n={n_short}), "
                            f"diff {diff_min:+.0f} min."
                        ),
                        direction="positive" if diff_min > 0 else "negative",
                        magnitude=abs(diff_min),
                    ))

    return findings


def _find_workout_type_sleep(df: pd.DataFrame) -> list[Finding]:
    findings = []
    if "workout_type" not in df.columns or "total_hours" not in df.columns:
        return findings
    sub = df[["workout_type", "total_hours", "deep_pct"]].dropna(subset=["workout_type", "total_hours"])
    if len(sub) < 6:
        return findings
    by_type = sub.groupby("workout_type").agg(
        avg_sleep=("total_hours", "mean"),
        avg_deep =("deep_pct",   "mean"),
        n        =("total_hours", "count"),
    ).query("n >= 2").sort_values("avg_sleep", ascending=False)
    if len(by_type) >= 2:
        best  = by_type.iloc[0]
        worst = by_type.iloc[-1]
        diff  = (best["avg_sleep"] - worst["avg_sleep"]) * 60
        if diff >= 10:
            findings.append(Finding(
                category="Sleep vs Workouts",
                title=f"{best.name.title()} gives the best sleep ({worst.name.title()} the worst)",
                detail=(
                    f"Best sleep after {best.name} (n={best['n']:.0f}): {best['avg_sleep']:.1f}h. "
                    f"Worst after {worst.name} (n={worst['n']:.0f}): {worst['avg_sleep']:.1f}h "
                    f"({diff:.0f} min difference)."
                ),
                direction="neutral",
                magnitude=diff,
            ))
    return findings


def _find_rhr_patterns(df: pd.DataFrame) -> list[Finding]:
    findings = []
    if "resting_hr" not in df.columns:
        return findings

    rhr = df["resting_hr"].dropna()
    if len(rhr) < 7:
        return findings

    # RHR trend: first vs second half
    mid = len(rhr) // 2
    early_rhr = rhr.iloc[:mid].mean()
    late_rhr  = rhr.iloc[mid:].mean()
    delta = late_rhr - early_rhr
    if abs(delta) >= 1.5:
        findings.append(Finding(
            category="Cardio & RHR",
            title=f"Resting HR {'improving' if delta < 0 else 'rising'}: {early_rhr:.0f} → {late_rhr:.0f} bpm",
            detail=(
                f"Your RHR moved from ~{early_rhr:.0f} bpm in the first half of the period "
                f"to ~{late_rhr:.0f} bpm in the second half ({delta:+.1f} bpm). "
                + ("A falling RHR signals improving cardiovascular fitness."
                   if delta < 0 else
                   "A rising RHR can indicate accumulated fatigue, illness, or reduced training.")
            ),
            direction="positive" if delta < 0 else "negative",
            magnitude=abs(delta),
        ))

    # RHR vs sleep duration
    r = _bucket_diff(df["total_hours"] if "total_hours" in df.columns else pd.Series(dtype=float),
                     df["resting_hr"], 6.5, 7.5)
    if r:
        rhr_short, rhr_long, n_short, n_long = r
        diff = rhr_long - rhr_short
        if abs(diff) >= 1:
            findings.append(Finding(
                category="Cardio & RHR",
                title=f"More sleep → {'lower' if diff < 0 else 'higher'} resting HR",
                detail=(
                    f"RHR after ≥7.5h sleep (n={n_long}): {rhr_long:.0f} bpm "
                    f"vs {rhr_short:.0f} bpm after ≤6.5h (n={n_short}), diff {diff:+.1f} bpm."
                ),
                direction="positive" if diff < 0 else "negative",
                magnitude=abs(diff),
            ))

    # RHR vs Garmin stress
    if "avg_stress" in df.columns:
        sub = df[["avg_stress", "resting_hr"]].dropna()
        if len(sub) >= 6:
            corr = sub["avg_stress"].corr(sub["resting_hr"])
            if abs(corr) >= 0.2:
                direction = "negative" if corr > 0 else "positive"
                findings.append(Finding(
                    category="Cardio & RHR",
                    title=f"Stress and resting HR are {'positively' if corr > 0 else 'negatively'} correlated (r={corr:.2f})",
                    detail=(
                        f"Days with higher Garmin stress scores tend to have "
                        f"{'higher' if corr > 0 else 'lower'} resting HR "
                        f"(Pearson r={corr:.2f}, n={len(sub)})."
                    ),
                    direction=direction,
                    magnitude=abs(corr) * 10,
                ))

    # Workout days vs RHR
    r = _mean_diff(df, "had_workout", "resting_hr")
    if r:
        with_wk, without_wk, n_with, n_without = r
        diff = with_wk - without_wk
        if abs(diff) >= 1:
            findings.append(Finding(
                category="Cardio & RHR",
                title=f"Workout days have {'lower' if diff < 0 else 'higher'} resting HR",
                detail=(
                    f"RHR on workout days: {with_wk:.0f} bpm vs {without_wk:.0f} bpm on rest days "
                    f"({diff:+.1f} bpm, n={n_with} workout days)."
                ),
                direction="positive" if diff < 0 else "neutral",
                magnitude=abs(diff),
            ))

    return findings


def _find_steps_patterns(df: pd.DataFrame) -> list[Finding]:
    findings = []
    if "total_steps" not in df.columns:
        return findings

    steps = df["total_steps"].dropna()
    if len(steps) < 7:
        return findings

    avg_steps = steps.mean()
    findings.append(Finding(
        category="Steps & Activity",
        title=f"Average daily steps: {avg_steps:,.0f}",
        detail=(
            f"You averaged {avg_steps:,.0f} steps/day over this period. "
            + ("Above the WHO 8,000/day recommendation." if avg_steps >= 8000 else
               "Below the WHO 8,000/day recommendation — even 1,000 extra steps/day reduces all-cause mortality.")
        ),
        direction="positive" if avg_steps >= 8000 else "negative",
        magnitude=abs(avg_steps - 8000) / 1000,
    ))

    # Steps vs same-day Garmin stress
    if "avg_stress" in df.columns:
        sub = df[["total_steps", "avg_stress"]].dropna()
        if len(sub) >= 6:
            corr = sub["total_steps"].corr(sub["avg_stress"])
            if abs(corr) >= 0.15:
                findings.append(Finding(
                    category="Steps & Activity",
                    title=f"More steps → {'less' if corr < 0 else 'more'} daily stress",
                    detail=(
                        f"Steps and Garmin stress are {'negatively' if corr < 0 else 'positively'} "
                        f"correlated (r={corr:.2f}, n={len(sub)}). "
                        + ("Physical activity appears to reduce your daily stress levels."
                           if corr < 0 else
                           "High step days tend to coincide with higher stress — check if these are busy, high-demand days.")
                    ),
                    direction="positive" if corr < 0 else "negative",
                    magnitude=abs(corr) * 10,
                ))

    # Steps vs body battery
    if "body_battery_highest" in df.columns:
        sub = df[["total_steps", "body_battery_highest"]].dropna()
        if len(sub) >= 6:
            corr = sub["total_steps"].corr(sub["body_battery_highest"])
            if abs(corr) >= 0.15:
                findings.append(Finding(
                    category="Steps & Activity",
                    title=f"Steps and body battery are {'positively' if corr > 0 else 'negatively'} correlated (r={corr:.2f})",
                    detail=(
                        f"Higher step counts are associated with "
                        f"{'higher' if corr > 0 else 'lower'} peak body battery "
                        f"(r={corr:.2f}, n={len(sub)})."
                    ),
                    direction="positive" if corr > 0 else "negative",
                    magnitude=abs(corr) * 10,
                ))

    # Steps vs resting HR
    if "resting_hr" in df.columns:
        sub = df[["total_steps", "resting_hr"]].dropna()
        if len(sub) >= 6:
            corr = sub["total_steps"].corr(sub["resting_hr"])
            if abs(corr) >= 0.15:
                findings.append(Finding(
                    category="Steps & Activity",
                    title=f"High-step days have {'lower' if corr < 0 else 'higher'} resting HR",
                    detail=(
                        f"Steps and resting HR are correlated at r={corr:.2f} (n={len(sub)}). "
                        + ("More movement correlates with a lower resting HR."
                           if corr < 0 else
                           "High step days seem to coincide with elevated resting HR — could be correlated with busy/stressful days.")
                    ),
                    direction="positive" if corr < 0 else "negative",
                    magnitude=abs(corr) * 10,
                ))

    # Sleep quality → next day steps
    if "next_total_steps" in df.columns and "total_hours" in df.columns:
        r = _bucket_diff(df["total_hours"], df["next_total_steps"], 6.5, 7.5)
        if r:
            short_steps, long_steps, n_short, n_long = r
            diff = long_steps - short_steps
            if abs(diff) >= 500:
                findings.append(Finding(
                    category="Steps & Activity",
                    title=f"Good sleep → {'more' if diff > 0 else 'fewer'} steps next day",
                    detail=(
                        f"Days following ≥7.5h sleep (n={n_long}): {long_steps:,.0f} steps avg "
                        f"vs {short_steps:,.0f} steps after ≤6.5h (n={n_short}), diff {diff:+,.0f}."
                    ),
                    direction="positive" if diff > 0 else "negative",
                    magnitude=abs(diff) / 100,
                ))

    return findings


def _find_recovery_patterns(df: pd.DataFrame) -> list[Finding]:
    findings = []

    # Body battery: sleep → peak BB next day
    if "next_body_battery_highest" in df.columns and "total_hours" in df.columns:
        r = _bucket_diff(df["total_hours"], df["next_body_battery_highest"], 6.5, 7.5)
        if r:
            bb_short, bb_long, n_short, n_long = r
            diff = bb_long - bb_short
            if abs(diff) >= 3:
                findings.append(Finding(
                    category="Recovery",
                    title=f"More sleep → {'higher' if diff > 0 else 'lower'} next-day body battery",
                    detail=(
                        f"Peak body battery after ≥7.5h sleep (n={n_long}): {bb_long:.0f} "
                        f"vs {bb_short:.0f} after <6.5h (n={n_short}), diff {diff:+.0f} points."
                    ),
                    direction="positive" if diff > 0 else "negative",
                    magnitude=abs(diff),
                ))

    # Body battery: workout → next day BB floor
    if "next_body_battery_lowest" in df.columns and "had_workout" in df.columns:
        r = _mean_diff(df, "had_workout", "next_body_battery_lowest")
        if r:
            with_wk, without_wk, n_with, n_without = r
            diff = with_wk - without_wk
            if abs(diff) >= 3:
                findings.append(Finding(
                    category="Recovery",
                    title=f"Day after workout: body battery floor is {'higher' if diff > 0 else 'lower'}",
                    detail=(
                        f"Battery floor the day after a workout (n={n_with}): {with_wk:.0f} "
                        f"vs {without_wk:.0f} after rest (n={n_without}), diff {diff:+.0f} points."
                    ),
                    direction="positive" if diff > 0 else "negative",
                    magnitude=abs(diff),
                ))

    # BB recovery (spread) vs sleep
    if "bb_recovery" in df.columns and "total_hours" in df.columns:
        sub = df[["total_hours", "bb_recovery"]].dropna()
        if len(sub) >= 6:
            corr = sub["total_hours"].corr(sub["bb_recovery"])
            if abs(corr) >= 0.15:
                findings.append(Finding(
                    category="Recovery",
                    title=f"More sleep → {'better' if corr > 0 else 'worse'} body battery recovery",
                    detail=(
                        f"Sleep duration and daily body battery swing correlate at r={corr:.2f} (n={len(sub)}). "
                        + ("Longer sleep leads to greater daily energy range."
                           if corr > 0 else
                           "Unexpected negative correlation — may reflect days with high sleep debt.")
                    ),
                    direction="positive" if corr > 0 else "negative",
                    magnitude=abs(corr) * 10,
                ))

    # Stress → next day body battery
    if "next_body_battery_highest" in df.columns and "avg_stress" in df.columns:
        sub = df[["avg_stress", "next_body_battery_highest"]].dropna()
        if len(sub) >= 6:
            corr = sub["avg_stress"].corr(sub["next_body_battery_highest"])
            if abs(corr) >= 0.15:
                findings.append(Finding(
                    category="Recovery",
                    title=f"High stress → {'lower' if corr < 0 else 'higher'} next-day body battery",
                    detail=(
                        f"Today's stress and tomorrow's peak body battery correlate at r={corr:.2f} (n={len(sub)}). "
                        + ("High stress days drain tomorrow's battery." if corr < 0 else "")
                    ),
                    direction="negative" if corr < 0 else "neutral",
                    magnitude=abs(corr) * 10,
                ))

    return findings


def _find_hrv_patterns(df: pd.DataFrame) -> list[Finding]:
    findings = []
    if "hrv_nightly_avg" not in df.columns:
        return findings

    hrv = df["hrv_nightly_avg"].dropna()
    if len(hrv) < 6:
        return findings

    # HRV trend
    mid = len(hrv) // 2
    early = hrv.iloc[:mid].mean()
    late  = hrv.iloc[mid:].mean()
    delta = late - early
    if abs(delta) >= 1:
        findings.append(Finding(
            category="HRV",
            title=f"Nightly HRV {'improving' if delta > 0 else 'declining'}: {early:.0f} → {late:.0f}",
            detail=(
                f"HRV moved from ~{early:.0f} in the first half to ~{late:.0f} in the second half ({delta:+.0f}). "
                + ("Rising HRV suggests better recovery and autonomic balance."
                   if delta > 0 else
                   "Declining HRV can signal accumulated fatigue or physiological stress.")
            ),
            direction="positive" if delta > 0 else "negative",
            magnitude=abs(delta),
        ))

    # HRV vs Garmin stress
    if "avg_stress" in df.columns:
        sub = df[["hrv_nightly_avg", "avg_stress"]].dropna()
        if len(sub) >= 6:
            corr = sub["hrv_nightly_avg"].corr(sub["avg_stress"])
            if abs(corr) >= 0.15:
                findings.append(Finding(
                    category="HRV",
                    title=f"HRV and daily stress are {'inversely' if corr < 0 else 'positively'} linked (r={corr:.2f})",
                    detail=(
                        f"Nights with higher HRV tend to follow days with {'lower' if corr < 0 else 'higher'} stress "
                        f"(r={corr:.2f}, n={len(sub)}). "
                        "A negative correlation is the healthy pattern — low stress → high HRV."
                    ),
                    direction="positive" if corr < 0 else "negative",
                    magnitude=abs(corr) * 10,
                ))

    # HRV vs sleep depth
    if "deep_pct" in df.columns:
        sub = df[["hrv_nightly_avg", "deep_pct"]].dropna()
        if len(sub) >= 6:
            corr = sub["hrv_nightly_avg"].corr(sub["deep_pct"])
            if abs(corr) >= 0.15:
                findings.append(Finding(
                    category="HRV",
                    title=f"Higher HRV nights have {'more' if corr > 0 else 'less'} deep sleep (r={corr:.2f})",
                    detail=(
                        f"Nightly HRV and deep sleep % correlate at r={corr:.2f} (n={len(sub)}). "
                        "Higher HRV typically co-occurs with deeper, more restorative sleep."
                    ),
                    direction="positive" if corr > 0 else "negative",
                    magnitude=abs(corr) * 10,
                ))

    # HRV vs alcohol (from feedback)
    if "had_alcohol" in df.columns:
        r = _mean_diff(df, "had_alcohol", "hrv_nightly_avg")
        if r:
            with_alc, without_alc, n_with, n_without = r
            diff = with_alc - without_alc
            if abs(diff) >= 1:
                findings.append(Finding(
                    category="HRV",
                    title=f"Alcohol → {'higher' if diff > 0 else 'lower'} HRV",
                    detail=(
                        f"HRV after alcohol (n={n_with}): {with_alc:.0f} "
                        f"vs {without_alc:.0f} sober nights ({diff:+.1f}). "
                        "Alcohol consistently lowers HRV even in small amounts."
                    ),
                    direction="negative" if diff < 0 else "neutral",
                    magnitude=abs(diff),
                ))

    return findings


def _find_stress_patterns(df: pd.DataFrame) -> list[Finding]:
    findings = []
    if "avg_stress" not in df.columns:
        return findings

    stress = df["avg_stress"].dropna()
    if len(stress) < 6:
        return findings

    avg_s = stress.mean()
    findings.append(Finding(
        category="Stress",
        title=f"Average Garmin stress: {avg_s:.0f}/100",
        detail=(
            f"Your mean daily stress score is {avg_s:.0f}. "
            + ("Scores above 40 suggest chronically elevated stress." if avg_s > 40 else
               "This is in a healthy range (below 40)." if avg_s < 40 else "")
        ),
        direction="negative" if avg_s > 40 else "positive",
        magnitude=abs(avg_s - 30),
    ))

    # Stress → sleep depth
    if "deep_pct" in df.columns:
        sub = df[["avg_stress", "deep_pct"]].dropna()
        if len(sub) >= 6:
            corr = sub["avg_stress"].corr(sub["deep_pct"])
            if abs(corr) >= 0.15:
                findings.append(Finding(
                    category="Stress",
                    title=f"High-stress days have {'less' if corr < 0 else 'more'} deep sleep",
                    detail=(
                        f"Garmin stress and deep sleep % correlate at r={corr:.2f} (n={len(sub)}). "
                        + ("Stress suppresses deep sleep." if corr < 0 else "")
                    ),
                    direction="negative" if corr < 0 else "neutral",
                    magnitude=abs(corr) * 10,
                ))

    # High vs low stress days: body battery drain
    if "bb_recovery" in df.columns:
        r = _bucket_diff(df["avg_stress"], df["bb_recovery"], 25, 50)
        if r:
            bb_low_stress, bb_hi_stress, n_lo, n_hi = r
            diff = bb_hi_stress - bb_low_stress
            findings.append(Finding(
                category="Stress",
                title=f"High-stress days drain {'more' if diff > 0 else 'less'} body battery",
                detail=(
                    f"On high-stress days (≥50, n={n_hi}): battery swing {bb_hi_stress:.0f} pts "
                    f"vs {bb_low_stress:.0f} pts on low-stress days (≤25, n={n_lo})."
                ),
                direction="negative" if diff > 0 else "positive",
                magnitude=abs(diff),
            ))

    return findings


def _find_sleep_quality_patterns(df: pd.DataFrame) -> list[Finding]:
    findings = []

    # Alcohol
    for metric, label in [("deep_pct", "deep sleep %"), ("rem_pct", "REM %"), ("total_hours", "total sleep")]:
        r = _mean_diff(df, "had_alcohol", metric)
        if not r:
            continue
        with_alc, without_alc, n_with, n_without = r
        diff = with_alc - without_alc
        if metric == "total_hours":
            diff *= 60
            unit = "min"
        else:
            unit = "pp"
        if abs(diff) < (8 if unit == "min" else 1.5):
            continue
        findings.append(Finding(
            category="Lifestyle & Sleep",
            title=f"Alcohol → {'more' if diff > 0 else 'less'} {label}",
            detail=(
                f"Nights with alcohol (n={n_with}): {with_alc:.1f} vs {without_alc:.1f} sober "
                f"({diff:+.0f}{unit}). Alcohol disrupts sleep architecture even in small amounts."
            ),
            direction="negative" if diff < 0 else "neutral",
            magnitude=abs(diff),
        ))

    # Late meal
    for metric, label in [("deep_pct", "deep sleep"), ("total_hours", "total sleep")]:
        r = _mean_diff(df, "late_meal", metric)
        if not r:
            continue
        with_lm, without_lm, n_with, n_without = r
        diff = with_lm - without_lm
        if metric == "total_hours":
            diff *= 60
            unit = "min"
        else:
            unit = "pp"
        if abs(diff) < (8 if unit == "min" else 1.5):
            continue
        findings.append(Finding(
            category="Lifestyle & Sleep",
            title=f"Late meal → {'more' if diff > 0 else 'less'} {label}",
            detail=(
                f"After late meals (n={n_with}): {with_lm:.1f} vs {without_lm:.1f} without "
                f"({diff:+.0f}{unit})."
            ),
            direction="negative" if diff < 0 else "neutral",
            magnitude=abs(diff),
        ))

    # Caffeine
    if "caffeine_after_2pm" in df.columns:
        r = _mean_diff(df, "caffeine_after_2pm", "deep_pct")
        if r:
            with_c, without_c, n_with, n_without = r
            diff = with_c - without_c
            if abs(diff) >= 1.5:
                findings.append(Finding(
                    category="Lifestyle & Sleep",
                    title=f"Caffeine after 2pm → {'more' if diff > 0 else 'less'} deep sleep",
                    detail=(
                        f"Deep sleep on caffeine nights (n={n_with}): {with_c:.1f}% "
                        f"vs {without_c:.1f}% without ({diff:+.1f} pp)."
                    ),
                    direction="negative" if diff < 0 else "neutral",
                    magnitude=abs(diff),
                ))

    return findings


def _find_spo2_patterns(df: pd.DataFrame) -> list[Finding]:
    findings = []
    sub = df["lowest_spo2"].dropna() if "lowest_spo2" in df.columns else pd.Series(dtype=float)
    if len(sub) < 5:
        return findings
    below_90 = (sub < 90).sum()
    below_94 = (sub < 94).sum()
    if below_90 > 0:
        findings.append(Finding(
            category="Sleep Quality",
            title="SpO2 dipped below 90% during sleep",
            detail=f"Lowest SpO2 fell below 90% on {below_90} night(s). Worth discussing with a doctor.",
            direction="negative", magnitude=10.0,
        ))
    elif below_94 >= 3:
        findings.append(Finding(
            category="Sleep Quality",
            title=f"SpO2 dropped below 94% on {below_94} nights",
            detail="Values below 94% during sleep may warrant attention.",
            direction="negative", magnitude=6.0,
        ))
    else:
        avg_spo2 = sub.mean()
        findings.append(Finding(
            category="Sleep Quality",
            title=f"SpO2 consistently good: avg {avg_spo2:.1f}%",
            detail=f"Avg SpO2 during sleep is {avg_spo2:.1f}% with no nights below 94% — healthy range.",
            direction="positive", magnitude=1.0,
        ))
    return findings


def _find_active_minutes(df: pd.DataFrame) -> list[Finding]:
    findings = []

    if "active_seconds" in df.columns:
        sub = df["active_seconds"].dropna()
        if len(sub) >= 6:
            avg_min = sub.mean() / 60
            weekly_min = avg_min * 7
            findings.append(Finding(
                category="Steps & Activity",
                title=f"Average {avg_min:.0f} active min/day ({weekly_min:.0f} min/week)",
                detail=(
                    f"You average {avg_min:.0f} active minutes per day. "
                    + ("Exceeds WHO minimum of 150 min/week of moderate activity." if weekly_min >= 150 else
                       f"Below the WHO 150 min/week target by {150 - weekly_min:.0f} min.")
                ),
                direction="positive" if weekly_min >= 150 else "negative",
                magnitude=abs(weekly_min - 150) / 10,
            ))

    # Highly active minutes vs next-day RHR
    if "highly_active_seconds" in df.columns and "next_resting_hr" in df.columns:
        sub = df[["highly_active_seconds", "next_resting_hr"]].dropna()
        if len(sub) >= 6:
            sub["vigorous_min"] = sub["highly_active_seconds"] / 60
            corr = sub["vigorous_min"].corr(sub["next_resting_hr"])
            if abs(corr) >= 0.15:
                findings.append(Finding(
                    category="Steps & Activity",
                    title=f"Vigorous activity → {'lower' if corr < 0 else 'higher'} next-day RHR",
                    detail=(
                        f"Highly active minutes and next-day resting HR correlate at r={corr:.2f} (n={len(sub)}). "
                        + ("More vigorous movement leads to lower resting HR." if corr < 0 else "")
                    ),
                    direction="positive" if corr < 0 else "neutral",
                    magnitude=abs(corr) * 10,
                ))

    return findings


# ── Plots ─────────────────────────────────────────────────────────────────────

def get_plots(days: int = 90) -> list[tuple[str, object]]:
    """Return list of (title, plotly_figure) for display in the report."""
    df = _load_combined(days)
    if df.empty:
        return []

    plots = []
    colours = {"workout": "#00B2A9", "rest": "#636EFA",
               "positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#F39C12"}

    # 1. Sleep duration: workout vs rest days
    if "had_workout" in df.columns and "total_hours" in df.columns:
        sub = df[["had_workout", "total_hours"]].dropna()
        if len(sub) >= 5:
            sub["Day type"] = sub["had_workout"].map({True: "Workout day", False: "Rest day"})
            fig = px.box(sub, x="Day type", y="total_hours",
                         color="Day type",
                         color_discrete_map={"Workout day": colours["workout"], "Rest day": colours["rest"]},
                         labels={"total_hours": "Sleep (hours)"})
            fig.update_layout(title="Sleep duration: workout vs rest days", showlegend=False, height=350)
            plots.append(("Sleep duration: workout vs rest days", fig))

    # 2. Deep sleep %: workout vs rest days
    if "had_workout" in df.columns and "deep_pct" in df.columns:
        sub = df[["had_workout", "deep_pct"]].dropna()
        if len(sub) >= 5:
            sub["Day type"] = sub["had_workout"].map({True: "Workout day", False: "Rest day"})
            fig = px.box(sub, x="Day type", y="deep_pct",
                         color="Day type",
                         color_discrete_map={"Workout day": colours["workout"], "Rest day": colours["rest"]},
                         labels={"deep_pct": "Deep sleep (%)"})
            fig.add_hline(y=20, line_dash="dash", line_color="gray", annotation_text="20% target")
            fig.update_layout(title="Deep sleep %: workout vs rest days", showlegend=False, height=350)
            plots.append(("Deep sleep %: workout vs rest days", fig))

    # 3. Sleep by workout type
    if "workout_type" in df.columns and "total_hours" in df.columns:
        sub = df[["workout_type", "total_hours", "deep_pct"]].dropna(subset=["workout_type", "total_hours"])
        counts = sub["workout_type"].value_counts()
        valid_types = counts[counts >= 2].index
        sub = sub[sub["workout_type"].isin(valid_types)]
        if len(sub) >= 5 and sub["workout_type"].nunique() >= 2:
            fig = px.box(sub, x="workout_type", y="total_hours",
                         labels={"workout_type": "Workout type", "total_hours": "Sleep (hours)"},
                         color="workout_type")
            fig.update_layout(title="Sleep duration by workout type", showlegend=False, height=350,
                              xaxis_tickangle=-30)
            plots.append(("Sleep duration by workout type", fig))

    # 4. Steps vs RHR scatter
    if "total_steps" in df.columns and "resting_hr" in df.columns:
        sub = df[["total_steps", "resting_hr", "date"]].dropna()
        if len(sub) >= 6:
            fig = px.scatter(sub, x="total_steps", y="resting_hr",
                             trendline="ols",
                             labels={"total_steps": "Daily steps", "resting_hr": "Resting HR (bpm)"},
                             color_discrete_sequence=[colours["workout"]])
            fig.update_layout(title="Steps vs resting heart rate", height=350)
            plots.append(("Steps vs resting HR", fig))

    # 5. Steps vs Garmin stress scatter
    if "total_steps" in df.columns and "avg_stress" in df.columns:
        sub = df[["total_steps", "avg_stress"]].dropna()
        if len(sub) >= 6:
            fig = px.scatter(sub, x="total_steps", y="avg_stress",
                             trendline="ols",
                             labels={"total_steps": "Daily steps", "avg_stress": "Garmin stress score"},
                             color_discrete_sequence=[colours["neutral"]])
            fig.update_layout(title="Steps vs daily stress", height=350)
            plots.append(("Steps vs daily stress", fig))

    # 6. Workout HR vs that night's sleep
    if "workout_avg_hr" in df.columns and "total_hours" in df.columns:
        sub = df[["workout_avg_hr", "total_hours", "deep_pct"]].dropna()
        if len(sub) >= 5:
            fig = px.scatter(sub, x="workout_avg_hr", y="total_hours",
                             color="deep_pct",
                             color_continuous_scale="Blues",
                             labels={"workout_avg_hr": "Workout avg HR (bpm)",
                                     "total_hours": "Sleep that night (h)",
                                     "deep_pct": "Deep %"},
                             trendline="ols")
            fig.update_layout(title="Workout intensity vs sleep that night", height=350)
            plots.append(("Workout HR vs sleep", fig))

    # 7. RHR trend over time
    if "resting_hr" in df.columns and "date" in df.columns:
        sub = df[["date", "resting_hr"]].dropna().sort_values("date")
        if len(sub) >= 7:
            sub["date"] = pd.to_datetime(sub["date"])
            sub["rhr_rolling"] = sub["resting_hr"].rolling(7, min_periods=3).mean()
            fig = go.Figure()
            fig.add_scatter(x=sub["date"], y=sub["resting_hr"],
                            mode="markers", name="Daily RHR",
                            marker=dict(color=colours["negative"], opacity=0.4, size=6))
            fig.add_scatter(x=sub["date"], y=sub["rhr_rolling"],
                            mode="lines", name="7-day avg",
                            line=dict(color=colours["negative"], width=2))
            fig.update_layout(title="Resting HR trend", yaxis_title="bpm", height=350)
            plots.append(("Resting HR trend", fig))

    # 8. HRV trend over time
    if "hrv_nightly_avg" in df.columns and "date" in df.columns:
        sub = df[["date", "hrv_nightly_avg"]].dropna().sort_values("date")
        if len(sub) >= 7:
            sub["date"] = pd.to_datetime(sub["date"])
            sub["hrv_rolling"] = sub["hrv_nightly_avg"].rolling(7, min_periods=3).mean()
            fig = go.Figure()
            fig.add_scatter(x=sub["date"], y=sub["hrv_nightly_avg"],
                            mode="markers", name="Nightly HRV",
                            marker=dict(color=colours["positive"], opacity=0.4, size=6))
            fig.add_scatter(x=sub["date"], y=sub["hrv_rolling"],
                            mode="lines", name="7-day avg",
                            line=dict(color=colours["positive"], width=2))
            fig.update_layout(title="HRV trend", yaxis_title="HRV", height=350)
            plots.append(("HRV trend", fig))

    # 9. Body battery by sleep length bracket
    if "total_hours" in df.columns and "body_battery_highest" in df.columns:
        sub = df[["total_hours", "body_battery_highest"]].dropna()
        if len(sub) >= 6:
            sub["Sleep bracket"] = pd.cut(
                sub["total_hours"],
                bins=[0, 6, 6.5, 7, 7.5, 99],
                labels=["<6h", "6–6.5h", "6.5–7h", "7–7.5h", ">7.5h"],
            )
            fig = px.box(sub, x="Sleep bracket", y="body_battery_highest",
                         labels={"body_battery_highest": "Peak body battery"},
                         color="Sleep bracket",
                         color_discrete_sequence=px.colors.sequential.Teal)
            fig.update_layout(title="Body battery by sleep duration", showlegend=False, height=350)
            plots.append(("Body battery by sleep duration", fig))

    # 10. HRV vs stress scatter
    if "hrv_nightly_avg" in df.columns and "avg_stress" in df.columns:
        sub = df[["hrv_nightly_avg", "avg_stress"]].dropna()
        if len(sub) >= 6:
            fig = px.scatter(sub, x="avg_stress", y="hrv_nightly_avg",
                             trendline="ols",
                             labels={"avg_stress": "Garmin stress score",
                                     "hrv_nightly_avg": "Nightly HRV"},
                             color_discrete_sequence=[colours["positive"]])
            fig.update_layout(title="HRV vs daily stress", height=350)
            plots.append(("HRV vs stress", fig))

    # 11. Workout duration vs sleep hours scatter
    if "workout_duration_min" in df.columns and "total_hours" in df.columns:
        sub = df[["workout_duration_min", "total_hours"]].dropna()
        if len(sub) >= 5:
            fig = px.scatter(sub, x="workout_duration_min", y="total_hours",
                             trendline="ols",
                             labels={"workout_duration_min": "Workout duration (min)",
                                     "total_hours": "Sleep that night (h)"},
                             color_discrete_sequence=[colours["workout"]])
            fig.update_layout(title="Workout duration vs sleep", height=350)
            plots.append(("Workout duration vs sleep", fig))

    # 12. Steps trend over time
    if "total_steps" in df.columns and "date" in df.columns:
        sub = df[["date", "total_steps"]].dropna().sort_values("date")
        if len(sub) >= 7:
            sub["date"] = pd.to_datetime(sub["date"])
            sub["steps_rolling"] = sub["total_steps"].rolling(7, min_periods=3).mean()
            fig = go.Figure()
            fig.add_bar(x=sub["date"], y=sub["total_steps"],
                        name="Daily steps", marker_color=colours["workout"], opacity=0.5)
            fig.add_scatter(x=sub["date"], y=sub["steps_rolling"],
                            mode="lines", name="7-day avg",
                            line=dict(color=colours["neutral"], width=2))
            fig.add_hline(y=8000, line_dash="dash", line_color="gray",
                          annotation_text="8,000 target")
            fig.update_layout(title="Daily steps trend", yaxis_title="Steps", height=350)
            plots.append(("Daily steps trend", fig))

    return plots


# ── Main entry points ─────────────────────────────────────────────────────────

def run_weekly_report(days: int = 90) -> list[Finding]:
    df = _load_combined(days)
    if df.empty:
        return []

    all_findings: list[Finding] = []
    all_findings += _find_workout_sleep(df)
    all_findings += _find_workout_type_sleep(df)
    all_findings += _find_rhr_patterns(df)
    all_findings += _find_steps_patterns(df)
    all_findings += _find_recovery_patterns(df)
    all_findings += _find_hrv_patterns(df)
    all_findings += _find_stress_patterns(df)
    all_findings += _find_sleep_quality_patterns(df)
    all_findings += _find_spo2_patterns(df)
    all_findings += _find_active_minutes(df)

    all_findings.sort(key=lambda f: f.magnitude, reverse=True)
    return all_findings
