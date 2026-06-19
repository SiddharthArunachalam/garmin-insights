"""
Garmin Insights — Streamlit web app.
Run with:  streamlit run src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import database as db
import garmin_sync
import insights
import correlations  # noqa: E402 — created alongside app.py

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Garmin Insights",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

# ── Auto-sync once per session ────────────────────────────────────────────────

if "synced_this_session" not in st.session_state:
    st.session_state.synced_this_session = False

if not st.session_state.synced_this_session:
    _msg = st.empty()

    def _auto_progress(msg: str):
        _msg.caption(f"Auto-syncing… {msg}")

    with st.spinner("Syncing Garmin data…"):
        _result = garmin_sync.sync_recent(days=90, progress_cb=_auto_progress)

    st.session_state.synced_this_session = True
    _msg.empty()
    if "error" in _result:
        st.error(f"Garmin sync failed: {_result['error']}")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⌚ Garmin Insights")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "😴 Sleep", "🏃 Workouts & Cardio", "📊 Weekly Report", "🤖 Ask Claude"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Sync Data")

    sync_days = st.slider("Days to sync", 7, 90, 30)

    if st.button("🔄 Sync Garmin", use_container_width=True):
        status = garmin_sync.get_connection_status()
        if not status["connected"]:
            st.error(f"Cannot connect to Garmin: {status.get('error', 'Unknown error')}")
        else:
            bar = st.progress(0, text="Connecting…")
            messages = []

            def _progress(msg):
                messages.append(msg)
                bar.progress(
                    min(len(messages) / max(sync_days, 1), 0.99),
                    text=msg,
                )

            counts = garmin_sync.sync_recent(days=sync_days, progress_cb=_progress)
            bar.progress(1.0, text="Done!")
            st.success(
                f"Synced: {counts.get('sleep', 0)} sleep nights, "
                f"{counts.get('summaries', 0)} daily summaries, "
                f"{counts.get('activities', 0)} activities"
            )
            st.rerun()

    st.divider()
    mn, mx = db.get_date_range_with_data()
    if mn and mx:
        st.caption(f"Data range: {mn} → {mx}")
    else:
        st.caption("No data synced yet")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sleep_df(start: str, end: str) -> pd.DataFrame:
    rows = db.get_sleep(start, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df["total_hours"] = df["total_seconds"] / 3600
    df["deep_hours"] = df["deep_seconds"] / 3600
    df["rem_hours"] = df["rem_seconds"] / 3600
    df["light_hours"] = df["light_seconds"] / 3600
    df["awake_min"] = df["awake_seconds"] / 60
    total = df["total_seconds"].replace(0, float("nan"))
    df["deep_pct"] = (df["deep_seconds"] / total * 100).round(1)
    df["rem_pct"] = (df["rem_seconds"] / total * 100).round(1)
    return df


def _daily_df(start: str, end: str) -> pd.DataFrame:
    rows = db.get_daily_summaries(start, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _activity_df(start: str, end: str) -> pd.DataFrame:
    rows = db.get_activities(start, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df["duration_min"] = (df["duration_seconds"] / 60).round(1)
    df["distance_km"] = (df["distance_meters"] / 1000).round(2)
    return df


def _feedback_df(start: str, end: str) -> pd.DataFrame:
    rows = db.get_sleep_feedback(start, end)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def date_range_picker(key: str, default_days: int = 30):
    c1, c2 = st.columns(2)
    end = date.today()
    start = end - timedelta(days=default_days)
    start = c1.date_input("From", value=start, key=f"{key}_start")
    end = c2.date_input("To", value=end, key=f"{key}_end")
    return start.isoformat(), end.isoformat()


def metric_card(label: str, value, delta=None, suffix: str = ""):
    st.metric(label, f"{value}{suffix}", delta=delta)


# ── Dashboard ─────────────────────────────────────────────────────────────────

def page_dashboard():
    st.header("Dashboard")

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=30)).isoformat()

    sleep_df = _sleep_df(start, end)
    daily_df = _daily_df(start, end)
    activity_df = _activity_df(start, end)

    if sleep_df.empty and daily_df.empty:
        st.info("No data yet. Use **Sync Garmin** in the sidebar to pull your data.")
        return

    # ── KPI row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        avg_sleep = sleep_df["total_hours"].mean() if not sleep_df.empty else None
        st.metric("Avg Sleep", f"{avg_sleep:.1f}h" if avg_sleep else "—")
    with c2:
        avg_deep = sleep_df["deep_pct"].mean() if not sleep_df.empty else None
        st.metric("Avg Deep %", f"{avg_deep:.0f}%" if avg_deep else "—", delta="20% target" if avg_deep and avg_deep < 20 else None)
    with c3:
        avg_rhr = daily_df["resting_hr"].mean() if not daily_df.empty else None
        st.metric("Avg Resting HR", f"{avg_rhr:.0f} bpm" if avg_rhr else "—")
    with c4:
        avg_stress = daily_df["avg_stress"].mean() if not daily_df.empty else None
        st.metric("Avg Stress", f"{avg_stress:.0f}/100" if avg_stress else "—")
    with c5:
        n_workouts = len(activity_df) if not activity_df.empty else 0
        st.metric("Workouts (30d)", n_workouts)

    st.divider()

    col_left, col_right = st.columns(2)

    # Sleep trend
    if not sleep_df.empty:
        with col_left:
            st.subheader("Sleep Duration (30 days)")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sleep_df["date"], y=sleep_df["deep_hours"], name="Deep", marker_color="#1E90FF"))
            fig.add_trace(go.Bar(x=sleep_df["date"], y=sleep_df["rem_hours"], name="REM", marker_color="#9B59B6"))
            fig.add_trace(go.Bar(x=sleep_df["date"], y=sleep_df["light_hours"], name="Light", marker_color="#85C1E9"))
            fig.add_hline(y=7, line_dash="dash", line_color="gray", annotation_text="7h target")
            fig.update_layout(barmode="stack", height=280, margin=dict(t=10, b=10), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # RHR + stress trend
    if not daily_df.empty:
        with col_right:
            st.subheader("Resting HR & Stress (30 days)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_df["date"], y=daily_df["resting_hr"],
                name="Resting HR", line=dict(color="#E74C3C"), yaxis="y1"
            ))
            fig.add_trace(go.Scatter(
                x=daily_df["date"], y=daily_df["avg_stress"],
                name="Avg Stress", line=dict(color="#F39C12", dash="dot"), yaxis="y2"
            ))
            fig.update_layout(
                height=280, margin=dict(t=10, b=10),
                yaxis=dict(title="HR (bpm)", side="left"),
                yaxis2=dict(title="Stress", side="right", overlaying="y"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Recent workouts table
    if not activity_df.empty:
        st.subheader("Recent Workouts")
        display = activity_df[["date", "name", "activity_type", "duration_min", "distance_km", "avg_hr", "calories"]].copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        display.columns = ["Date", "Name", "Type", "Duration (min)", "Distance (km)", "Avg HR", "Calories"]
        st.dataframe(display.head(10), use_container_width=True, hide_index=True)


# ── Sleep page ────────────────────────────────────────────────────────────────

def page_sleep():
    st.header("😴 Sleep Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Charts", "📝 Daily Feedback", "🤖 Sleep Insights"])

    with tab1:
        start, end = date_range_picker("sleep", 30)
        sleep_df = _sleep_df(start, end)
        feedback_df = _feedback_df(start, end)

        if sleep_df.empty:
            st.info("No sleep data for this range. Sync Garmin first.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Avg Total Sleep", f"{sleep_df['total_hours'].mean():.1f}h")
            with c2:
                st.metric("Avg Deep Sleep", f"{sleep_df['deep_pct'].mean():.0f}%")
            with c3:
                st.metric("Avg REM Sleep", f"{sleep_df['rem_pct'].mean():.0f}%")
            with c4:
                avg_spo2 = sleep_df["avg_spo2"].dropna().mean()
                st.metric("Avg SpO2", f"{avg_spo2:.1f}%" if not pd.isna(avg_spo2) else "—")

            # Sleep stages stacked bar
            st.subheader("Sleep Stages")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sleep_df["date"], y=sleep_df["deep_hours"], name="Deep", marker_color="#1E90FF"))
            fig.add_trace(go.Bar(x=sleep_df["date"], y=sleep_df["rem_hours"], name="REM", marker_color="#9B59B6"))
            fig.add_trace(go.Bar(x=sleep_df["date"], y=sleep_df["light_hours"], name="Light", marker_color="#85C1E9"))
            fig.add_trace(go.Bar(x=sleep_df["date"], y=sleep_df["awake_min"] / 60, name="Awake", marker_color="#E74C3C"))
            fig.add_hline(y=7, line_dash="dash", line_color="gray", annotation_text="7h target")
            fig.update_layout(barmode="stack", height=320, xaxis_title="Date", yaxis_title="Hours")
            st.plotly_chart(fig, use_container_width=True)

            col_l, col_r = st.columns(2)

            with col_l:
                st.subheader("SpO2 During Sleep")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=sleep_df["date"], y=sleep_df["avg_spo2"], name="Avg SpO2", line=dict(color="#00B2A9")))
                fig2.add_trace(go.Scatter(x=sleep_df["date"], y=sleep_df["lowest_spo2"], name="Lowest SpO2", line=dict(color="#E74C3C", dash="dot")))
                fig2.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="95% threshold")
                fig2.update_layout(height=260, yaxis_title="SpO2 %")
                st.plotly_chart(fig2, use_container_width=True)

            with col_r:
                st.subheader("Sleep HR & HRV")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=sleep_df["date"], y=sleep_df["avg_hr_during_sleep"], name="Avg HR", line=dict(color="#E74C3C")))
                if sleep_df["hrv_nightly_avg"].notna().any():
                    fig3.add_trace(go.Scatter(x=sleep_df["date"], y=sleep_df["hrv_nightly_avg"], name="HRV", line=dict(color="#2ECC71"), yaxis="y2"))
                    fig3.update_layout(yaxis2=dict(side="right", overlaying="y", title="HRV (ms)"))
                fig3.update_layout(height=260, yaxis_title="HR (bpm)")
                st.plotly_chart(fig3, use_container_width=True)

            # Subjective overlay
            if not feedback_df.empty:
                st.subheader("Garmin Data vs Your Subjective Ratings")
                merged = sleep_df.merge(feedback_df, on="date", how="left")
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=merged["date"], y=merged["total_hours"], name="Sleep Duration (h)", line=dict(color="#00B2A9")))
                if "subjective_quality" in merged.columns:
                    fig4.add_trace(go.Scatter(x=merged["date"], y=merged["subjective_quality"], name="Subjective Quality /10", line=dict(color="#F39C12", dash="dot"), yaxis="y2"))
                fig4.update_layout(
                    height=280,
                    yaxis=dict(title="Hours"),
                    yaxis2=dict(title="Rating /10", side="right", overlaying="y", range=[0, 10]),
                )
                st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        st.subheader("How did you sleep?")
        st.caption("Log nightly feedback to help Claude find patterns that Garmin can't detect alone.")

        today = date.today().isoformat()
        existing = db.get_feedback_for_date(today)

        with st.form("sleep_feedback_form"):
            d = st.date_input("Night of", value=date.today())
            col1, col2 = st.columns(2)
            with col1:
                quality = st.slider("Sleep quality (1=terrible, 10=great)", 1, 10, int(existing["subjective_quality"]) if existing else 5)
                energy = st.slider("Energy on waking (1=exhausted, 10=great)", 1, 10, int(existing["energy_on_waking"]) if existing else 5)
                stress = st.slider("Yesterday's stress level (1=none, 10=extreme)", 1, 10, int(existing["stress_level"]) if existing else 3)
            with col2:
                alcohol = st.number_input("Alcohol drinks last night", 0, 20, int(existing["alcohol_drinks"]) if existing else 0)
                late_meal = st.radio("Late meal (after 8pm)?", [0, 1], format_func=lambda x: "Yes" if x else "No", index=int(existing["late_meal"]) if existing else 0)
                caffeine = st.radio("Caffeine after 2pm?", [0, 1], format_func=lambda x: "Yes" if x else "No", index=int(existing["caffeine_after_2pm"]) if existing else 0)
            notes = st.text_area("Notes (e.g. stressed, travel, sick)", value=existing["notes"] if existing and existing["notes"] else "")

            if st.form_submit_button("💾 Save Feedback", use_container_width=True):
                db.upsert_sleep_feedback({
                    "date": d.isoformat(),
                    "subjective_quality": quality,
                    "energy_on_waking": energy,
                    "notes": notes,
                    "alcohol_drinks": alcohol,
                    "late_meal": late_meal,
                    "stress_level": stress,
                    "caffeine_after_2pm": caffeine,
                })
                st.success("Feedback saved!")
                st.rerun()

        # Show recent feedback
        fb_df = _feedback_df(
            (date.today() - timedelta(days=14)).isoformat(),
            date.today().isoformat()
        )
        if not fb_df.empty:
            st.subheader("Recent Feedback")
            display = fb_df[["date", "subjective_quality", "energy_on_waking", "stress_level", "alcohol_drinks", "notes"]].copy()
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")
            display.columns = ["Date", "Quality /10", "Energy /10", "Stress /10", "Alcohol", "Notes"]
            st.dataframe(display, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("AI Sleep Analysis")
        insight_days = st.slider("Analyse last N days", 7, 60, 14, key="sleep_insight_days")

        if st.button("🔍 Generate Sleep Insights", use_container_width=True):
            with st.container():
                placeholder = st.empty()
                full_text = ""
                for chunk in insights.generate_sleep_insights(insight_days):
                    full_text += chunk
                    placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)


# ── Workouts & Cardio page ────────────────────────────────────────────────────

def page_workouts():
    st.header("🏃 Workouts & Cardio")

    tab1, tab2 = st.tabs(["📊 Charts", "🤖 Cardio Insights"])

    with tab1:
        start, end = date_range_picker("workouts", 60)
        activity_df = _activity_df(start, end)
        daily_df = _daily_df(start, end)

        if activity_df.empty:
            st.info("No workout data for this range.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Workouts", len(activity_df))
            with c2:
                st.metric("Avg Duration", f"{activity_df['duration_min'].mean():.0f} min")
            with c3:
                avg_hr = activity_df["avg_hr"].dropna().mean()
                st.metric("Avg HR", f"{avg_hr:.0f} bpm" if not pd.isna(avg_hr) else "—")
            with c4:
                total_dist = activity_df["distance_km"].sum()
                st.metric("Total Distance", f"{total_dist:.1f} km")

            # Activity frequency by type
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Activity Types")
                type_counts = activity_df["activity_type"].value_counts().reset_index()
                type_counts.columns = ["Type", "Count"]
                fig = px.pie(type_counts, names="Type", values="Count", color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(height=280, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                st.subheader("Workout Duration Over Time")
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=activity_df["date"], y=activity_df["duration_min"],
                    name="Duration", marker_color="#00B2A9",
                    text=activity_df["name"], hovertemplate="%{text}<br>%{y:.0f} min"
                ))
                fig2.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="Minutes")
                st.plotly_chart(fig2, use_container_width=True)

            # HR during workouts
            st.subheader("Heart Rate During Workouts")
            hr_df = activity_df.dropna(subset=["avg_hr"])
            if not hr_df.empty:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=hr_df["date"], y=hr_df["avg_hr"], mode="markers+lines", name="Avg HR", line=dict(color="#E74C3C")))
                fig3.add_trace(go.Scatter(x=hr_df["date"], y=hr_df["max_hr"], mode="markers", name="Max HR", marker=dict(color="#F39C12", symbol="triangle-up")))
                fig3.update_layout(height=280, yaxis_title="HR (bpm)")
                st.plotly_chart(fig3, use_container_width=True)

        # Resting HR + VO2 max trend
        if not daily_df.empty:
            st.subheader("Resting Heart Rate Trend")
            rhr_df = daily_df.dropna(subset=["resting_hr"])
            if not rhr_df.empty:
                fig4 = px.line(rhr_df, x="date", y="resting_hr", color_discrete_sequence=["#E74C3C"])
                fig4.update_layout(height=240, yaxis_title="RHR (bpm)")
                st.plotly_chart(fig4, use_container_width=True)

        # Body battery
        if not daily_df.empty and daily_df["body_battery_highest"].notna().any():
            st.subheader("Body Battery")
            bb_df = daily_df.dropna(subset=["body_battery_highest"])
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=bb_df["date"], y=bb_df["body_battery_highest"], name="Highest", fill="tozeroy", line=dict(color="#2ECC71")))
            fig5.add_trace(go.Scatter(x=bb_df["date"], y=bb_df["body_battery_lowest"], name="Lowest", line=dict(color="#E74C3C")))
            fig5.update_layout(height=260, yaxis_title="Body Battery", yaxis_range=[0, 100])
            st.plotly_chart(fig5, use_container_width=True)

    with tab2:
        st.subheader("AI Cardio & Fitness Analysis")
        insight_days = st.slider("Analyse last N days", 14, 90, 30, key="cardio_insight_days")

        if st.button("🔍 Generate Cardio Insights", use_container_width=True):
            placeholder = st.empty()
            full_text = ""
            for chunk in insights.generate_cardio_insights(insight_days):
                full_text += chunk
                placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)


# ── Ask Claude page ───────────────────────────────────────────────────────────

def page_ask_claude():
    st.header("🤖 Ask Claude")
    st.caption("Ask anything about your health data. Claude has access to your sleep, workouts, stress, and body battery.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    context_days = st.slider("Context window (days of data)", 7, 90, 30, key="chat_context")

    # Suggested questions
    suggestions = [
        "What's affecting my sleep the most?",
        "How has my cardio fitness trended over the past month?",
        "Which days did I sleep worst and what happened before those nights?",
        "Am I overtraining or undertraining based on my recovery?",
        "How does my stress level affect my resting heart rate?",
    ]
    st.write("**Quick questions:**")
    cols = st.columns(len(suggestions))
    for i, q in enumerate(suggestions):
        if cols[i].button(q, key=f"suggest_{i}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    # Chat display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle suggested question
    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask about your health data…") or pending

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                for chunk in insights.ask(question, st.session_state.chat_history[:-1], context_days):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error: {e}\n\nMake sure `GEMINI_API_KEY` is set in your `.env` file."
                placeholder.markdown(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ── Weekly Report page ────────────────────────────────────────────────────────

def page_weekly_report():
    st.header("📊 Weekly Report")
    st.caption(
        "Automatically finds correlations across sleep, cardio, steps, HRV, stress, and recovery — "
        "no AI API needed. Sync at least 30 days for best results."
    )

    analysis_days = st.slider("Analyse last N days", 30, 180, 90, key="report_days")

    if st.button("🔍 Run Analysis", use_container_width=True):
        with st.spinner("Crunching your data…"):
            findings = correlations.run_weekly_report(analysis_days)
            plots    = correlations.get_plots(analysis_days)
        st.session_state["last_findings"]    = findings
        st.session_state["last_plots"]       = plots
        st.session_state["last_report_days"] = analysis_days

    findings    = st.session_state.get("last_findings")
    plots       = st.session_state.get("last_plots", [])
    report_days = st.session_state.get("last_report_days", analysis_days)

    if findings is None:
        st.info("Click **Run Analysis** to generate your report.")
        return

    if not findings:
        st.warning(
            "Not enough data to find patterns yet. "
            "Sync more Garmin data (90+ days recommended)."
        )
    else:
        st.success(f"Found {len(findings)} patterns across the last {report_days} days.")
        st.divider()

        icon = {"positive": "✅", "negative": "⚠️", "neutral": "ℹ️"}
        by_cat: dict[str, list] = {}
        for f in findings:
            by_cat.setdefault(f.category, []).append(f)

        for cat, cat_findings in by_cat.items():
            st.subheader(cat)
            for f in cat_findings:
                with st.container(border=True):
                    col_icon, col_text = st.columns([0.05, 0.95])
                    with col_icon:
                        st.write(icon.get(f.direction, "ℹ️"))
                    with col_text:
                        st.markdown(f"**{f.title}**")
                        st.markdown(f.detail)

    # ── Plots section ─────────────────────────────────────────────────────────
    if plots:
        st.divider()
        st.subheader("Charts")
        cols = st.columns(2)
        for i, (title, fig) in enumerate(plots):
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)


# ── Router ─────────────────────────────────────────────────────────────────────

if page == "🏠 Dashboard":
    page_dashboard()
elif page == "😴 Sleep":
    page_sleep()
elif page == "🏃 Workouts & Cardio":
    page_workouts()
elif page == "📊 Weekly Report":
    page_weekly_report()
elif page == "🤖 Ask Claude":
    page_ask_claude()
