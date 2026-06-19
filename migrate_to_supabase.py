"""
One-time migration: copies all data from local SQLite → Supabase (PostgreSQL).
Run once from your local machine after adding DATABASE_URL to .env.
"""

import sys
from pathlib import Path

sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv(".env")

import os
import sqlite3
import psycopg2
import psycopg2.extras

DB_PATH  = Path("data/garmin_insights.db")
DB_URL   = os.getenv("DATABASE_URL")

if not DB_URL:
    print("ERROR: DATABASE_URL not set in .env")
    sys.exit(1)

if not DB_PATH.exists():
    print(f"ERROR: SQLite DB not found at {DB_PATH}")
    sys.exit(1)

print(f"Connecting to Supabase...")
try:
    pg = psycopg2.connect(DB_URL)
    print("Connected.")
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

# Create schema on Supabase
import database as db
db.init_db()
print("Schema ready on Supabase.")

# Read everything from SQLite
sqlite = sqlite3.connect(DB_PATH)
sqlite.row_factory = sqlite3.Row

tables = {
    "sleep_records":   db._SLEEP_COLS,
    "daily_summaries": db._DAILY_COLS,
    "activities":      db._ACT_COLS,
    "fitness_age":     db._FITNESS_COLS,
    "sleep_feedback":  db._FEEDBACK_COLS,
    "calendar_events": db._CAL_COLS,
}

for table, cols in tables.items():
    rows = sqlite.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        print(f"  {table}: 0 rows (skipping)")
        continue

    # Build rows as dicts, only keeping columns that exist in SQLite
    sqlite_cols = [d[0] for d in sqlite.execute(f"SELECT * FROM {table} LIMIT 0").description]
    dicts = []
    for row in rows:
        d = dict(row)
        # Fill missing columns with None
        dicts.append({c: d.get(c) for c in cols})

    sql = db._upsert(table, cols[0], cols)  # cols[0] is always the PK
    with pg.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, dicts)
    pg.commit()
    print(f"  {table}: {len(dicts)} rows migrated")

sqlite.close()
pg.close()
print("\nDone. Your Supabase DB is ready.")
