from db.db_connect import get_connection
import pandas as pd

PIP = 0.0001


def load_data() -> pd.DataFrame:
    conn = get_connection()

    df = pd.read_sql(
        """
        SELECT open, high, low, close
        FROM market_data
        WHERE symbol = 'EURUSD' AND timeframe = '1D'
        """,
        conn,
    )

    conn.close()
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[["open", "high", "low", "close"]] = df[
        ["open", "high", "low", "close"]
    ].apply(pd.to_numeric)

    df["bull_sweep"] = (df["open"] - df["low"]) / PIP
    df["bear_sweep"] = (df["high"] - df["open"]) / PIP

    return df


def analyze_sweeps(df: pd.DataFrame) -> dict:
    bull = df[(df["bull_sweep"] > 5) & (df["bull_sweep"] <= 16)]
    bear = df[(df["bear_sweep"] > 5) & (df["bear_sweep"] <= 16)]

    bull_up = bull[bull["close"] > bull["open"]]
    bear_down = bear[bear["close"] < bear["open"]]

    return {
        "bull_total": len(bull),
        "bull_up": len(bull_up),
        "bear_total": len(bear),
        "bear_down": len(bear_down),
    }


def main():
    df = load_data()
    df = preprocess(df)

    stats = analyze_sweeps(df)

    print("------ 6–16 Sweep Analysis ------")
    print(f"Bullish sweeps: {stats['bull_total']}")
    print(f"→ Closed UP: {stats['bull_up']}")
    print()
    print(f"Bearish sweeps: {stats['bear_total']}")
    print(f"→ Closed DOWN: {stats['bear_down']}")


if __name__ == "__main__":
    main()
