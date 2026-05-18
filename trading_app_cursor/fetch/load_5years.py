from db.db_connect import get_connection
from config.config import API_KEY
from pathlib import Path
import sys
from datetime import date

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_5_years() -> None:
    url = (
        "https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol=EUR&to_symbol=USD"
        f"&outputsize=full&apikey={API_KEY}"
    )

    response = requests.get(url, timeout=120)

    if response.status_code != 200:
        print("API request failed:", response.status_code)
        return

    data = response.json()

    if "Time Series FX (Daily)" not in data:
        print("Invalid API response")
        return

    series = data["Time Series FX (Daily)"]

    conn = get_connection()
    cur = conn.cursor()

    inserted = 0

    for ts, v in series.items():

        if ts < "2019-01-01":
            continue

        day = date.fromisoformat(ts)

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

        inserted += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"Inserted {inserted} rows")


if __name__ == "__main__":
    load_5_years()
