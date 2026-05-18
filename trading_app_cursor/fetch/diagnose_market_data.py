from __future__ import annotations
from db.db_connect import get_connection
from config.config import API_KEY

import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fetch_av_series(outputsize: str) -> dict | None:
    url = (
        "https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol=EUR&to_symbol=USD"
        f"&outputsize={outputsize}&apikey={API_KEY}"
    )

    for _ in range(3):
        r = requests.get(url, timeout=90)
        data = r.json()

        if "Time Series FX (Daily)" in data:
            return data

        if "Note" in data or "Information" in data:
            time.sleep(65)
            continue

        return None

    return None


def get_db_stats():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(*), MAX(timestamp), MIN(timestamp)
        FROM market_data
        WHERE symbol = %s AND timeframe = %s;
        """,
        ("EURUSD", "1D"),
    )

    result = cur.fetchone()
    cur.close()
    conn.close()

    return result


def main() -> None:
    count, max_ts, min_ts = get_db_stats()

    print("DB Stats")
    print("Rows:", count)
    print("Min:", min_ts)
    print("Max:", max_ts)

    data = fetch_av_series("compact")
    if not data:
        sys.exit(1)

    keys = sorted(data["Time Series FX (Daily)"].keys(), reverse=True)
    api_latest = date.fromisoformat(keys[0])
    today = datetime.now(timezone.utc).date()

    print("\nAPI Stats")
    print("Latest dates:", keys[:5])
    print("API latest:", api_latest)
    print("UTC today:", today)

    if max_ts:
        db_latest = max_ts.date() if isinstance(
            max_ts, datetime) else date.fromisoformat(str(max_ts)[:10])

        print("\nComparison")

        if api_latest > db_latest:
            print("API is ahead of DB → run update script")
        elif api_latest == db_latest:
            print("DB is up to date with API")
        else:
            print("DB is ahead of API (check data source)")

    if api_latest < today:
        print("\nNote: API may not include latest daily bar yet.")


if __name__ == "__main__":
    main()
