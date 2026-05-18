import numpy as np
import pandas as pd

PIP = 0.0001


def _zones(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {
            "z0_5": 0,
            "z6_16": 0,
            "z17_1sd": 0,
            "z1sd_2sd": 0,
            "mean": 0,
            "std": 0,
            "total": 0,
        }

    mean = np.mean(arr)
    std = np.std(arr)

    return {
        "z0_5": int(((arr >= 0) & (arr <= 5)).sum()),
        "z6_16": int(((arr > 5) & (arr <= 16)).sum()),
        "z17_1sd": int(((arr > 16) & (arr <= mean + std)).sum()),
        "z1sd_2sd": int(((arr > mean + std) & (arr <= mean + 2 * std)).sum()),
        "mean": round(mean, 2),
        "std": round(std, 2),
        "total": len(arr),
    }


def liquidity_zone_analysis(df: pd.DataFrame) -> dict:
    df = df.copy()

    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(
        pd.to_numeric
    )

    body_pips = (df["close"] - df["open"]) / PIP

    bullish = df[body_pips >= 30]
    bearish = df[body_pips <= -30]

    bull_sweep = (bullish["open"] - bullish["low"]) / PIP
    bear_sweep = (bearish["high"] - bearish["open"]) / PIP

    bull = _zones(bull_sweep.to_numpy())
    bear = _zones(bear_sweep.to_numpy())

    return {
        "bull_0_5": bull["z0_5"],
        "bull_6_16": bull["z6_16"],
        "bull_17_1sd": bull["z17_1sd"],
        "bull_1sd_2sd": bull["z1sd_2sd"],
        "bull_mean": bull["mean"],
        "bull_total": bull["total"],

        "bear_0_5": bear["z0_5"],
        "bear_6_16": bear["z6_16"],
        "bear_17_1sd": bear["z17_1sd"],
        "bear_1sd_2sd": bear["z1sd_2sd"],
        "bear_mean": bear["mean"],
        "bear_total": bear["total"],
    }
