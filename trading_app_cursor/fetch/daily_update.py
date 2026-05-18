from __future__ import annotations
from db.db_connect import get_connection
from config.config import API_KEY

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fetch_series(outputsize: str, max_retries: int = 4) -> dict | None:
    if outputsize not in ("compact", "full"):
        raise ValueError("outputsize must be 'compact' or 'full'")

    url = (
        "https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol=EUR&to_symbol=USD"
        f"&outputsize={outputsize}&apikey={API_KEY}"
    )

    for attempt in range(max_retries):
        resp = requests.get(url, timeout=120)

        if resp.status_code != 200:
            print("HTTP error:", resp.status_code)
            return None

        data = resp.json()

        if "Error Message" in data:
            print("API error:", data["Error Message"])
            return None

        if "Note" in data or "Information" in data:
            wait = 65
            print(f"Rate limit hit (attempt {attempt + 1})")

            if attempt + 1 < max_retries:
                time.sleep(wait)
                continue

            return None

        if "Time Series FX (Daily)" not in data:
            print("Invalid response format")
            return None

        return data

    return None


def get_latest_db_timestamp(cur) -> str | None:
    cur.execute(
        """
        SELECT MAX(timestamp)::text
        FROM market_data
        WHERE symbol = %s AND timeframe = %s;
        """,
        ("EURUSD", "1D"),
    )

    row = cur.fetchone()
    return row[0] if row and row[0] else None


def update_missing_days(*, use_full: bool = False) -> int:
    print("Updating EURUSD data...")

    data = fetch_series("full" if use_full else "compact")
    if not data:
        return -1

    ts = data["Time Series FX (Daily)"]

    conn = get_connection()
    cur = conn.cursor()

    print("DB latest before update:", get_latest_db_timestamp(cur))

    updated = 0

    for d, v in ts.items():
        day = date.fromisoformat(d)

        cur.execute(
            """
            INSERT INTO market_data
            (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, timeframe, timestamp)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume;
            """,
            (
                "EURUSD",
                "1D",
                day,
                float(v["1. open"]),
                float(v["2. high"]),
                float(v["3. low"]),
                float(v["4. close"]),
                0,
            ),
        )

        updated += 1

    conn.commit()

    print("DB latest after update:", get_latest_db_timestamp(cur))

    cur.close()
    conn.close()

    print(f"Updated {updated} rows.")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")

    args = parser.parse_args()
    result = update_missing_days(use_full=args.full)

    if result < 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
