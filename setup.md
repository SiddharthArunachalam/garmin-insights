# Garmin Insights — Setup Guide

## 1. Install dependencies

```
pip install -r requirements.txt
```

## 2. Create your `.env` file

Copy `.env.example` to `.env` and fill in your credentials:

```
GARMIN_EMAIL=your_garmin_email@example.com
GARMIN_PASSWORD=your_garmin_password
GEMINI_API_KEY=AIza...
```

## 3. (Optional) Connect Google Calendar

a. Go to [Google Cloud Console](https://console.cloud.google.com)
b. Create a project → enable the **Google Calendar API**
c. Create OAuth 2.0 credentials (Desktop app) → download as `data/google_credentials.json`
d. Run the auth flow once:
   ```
   python src/calendar_sync.py --auth
   ```
   A browser window will open. Approve access. The token is saved automatically.

## 4. Run the app

```
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

## 5. Sync your data

- Click **Sync Garmin** in the sidebar to pull the last 30 days of data
- Click **Sync Calendar** (if set up) to pull calendar events
- Go to **Sleep → Daily Feedback** to start logging nightly notes

## Data stored locally

All data is stored in `data/garmin_insights.db` (SQLite). Nothing is sent anywhere except to:
- Garmin Connect (to pull your data)
- Gemini API (only when you click Generate Insights or send a chat message)
