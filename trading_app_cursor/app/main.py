from __future__ import annotations
from indicators.pip_stats import liquidity_zone_analysis
from indicators.next_day_candle import (
    candle_type_from_open_close,
    evaluate_next_day_model,
    fit_predict_next_day,
)
from db.db_connect import get_connection

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def format_candle_label(name: str) -> str:
    mapping = {
        "strong_bull": "Strong bullish (≥ 30 pips)",
        "strong_bear": "Strong bearish (≤ -30 pips)",
        "bull": "Mild bullish",
        "bear": "Mild bearish / flat",
    }
    return mapping.get(name, name)


st.set_page_config(page_title="Trading Dashboard", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "main"


def load_data() -> pd.DataFrame:
    conn = get_connection()

    query = """
        SELECT timestamp, open, high, low, close
        FROM market_data
        WHERE symbol='EURUSD' AND timeframe='1D'
        ORDER BY timestamp DESC;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_latest_timestamp() -> pd.Timestamp | None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT MAX(timestamp)
        FROM market_data
        WHERE symbol = %s AND timeframe = %s;
        """,
        ("EURUSD", "1D"),
    )

    row = cur.fetchone()
    cur.close()
    conn.close()

    return pd.to_datetime(row[0]) if row and row[0] else None


@st.cache_data(ttl=900)
def cached_forecast(data: pd.DataFrame) -> dict:
    return fit_predict_next_day(data)


def render_forecast(data: pd.DataFrame) -> None:
    st.subheader("Next Day Forecast")

    try:
        out = cached_forecast(data)
    except Exception as e:
        st.warning(str(e))
        return

    if not out.get("ok"):
        st.info(out.get("error", "Forecast failed"))
        return

    if out.get("last_bar_date"):
        st.caption(f"Latest completed daily bar: {out['last_bar_date']}")

    if out.get("forecast_target_date"):
        st.caption(f"Forecast target: {out['forecast_target_date']}")

    st.metric(
        "Predicted class",
        format_candle_label(out["predicted_class"]),
    )

    probs = out["probs"].sort_values()
    st.bar_chart(
        pd.DataFrame(
            {
                "probability": probs.values,
                "class": [format_candle_label(x) for x in probs.index],
            }
        ).set_index("class")
    )

    with st.expander("Raw probabilities"):
        st.write(out["probs"])

    with st.expander("Backtest"):
        if st.button("Run evaluation"):
            ev = evaluate_next_day_model(data)

            if not ev.get("ok"):
                st.error(ev.get("error"))
                return

            st.metric("Accuracy", f"{ev['test_accuracy_top1']:.1%}")
            st.metric(
                "Baseline", f"{ev['baseline_always_predict_train_mode']:.1%}")
            st.write(f"Log loss: {ev['multiclass_log_loss']:.3f}")
            st.write(f"Brier score: {ev['multiclass_brier_score']:.3f}")


df = load_data()
latest = get_latest_timestamp()

st.sidebar.header("Database")
st.sidebar.write("Latest:", latest.date() if latest else "N/A")

if st.sidebar.button("Refresh"):
    cached_forecast.clear()
    st.rerun()


# MAIN PAGE
if st.session_state.page == "main":

    st.title("EUR/USD Dashboard")

    st.dataframe(df, use_container_width=True)

    render_forecast(df)

    if st.button("Analysis Page"):
        st.session_state.page = "analysis"
        st.rerun()


# ANALYSIS PAGE
else:

    st.title("Liquidity Analysis")

    if st.button("Back"):
        st.session_state.page = "main"
        st.rerun()

    stats = liquidity_zone_analysis(df)

    render_forecast(df)

    st.subheader("Bullish Zones")
    st.write(stats["bull_mean"], stats["bull_total"])

    st.subheader("Bearish Zones")
    st.write(stats["bear_mean"], stats["bear_total"])
