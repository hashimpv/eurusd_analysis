from __future__ import annotations

import pandas as pd


def bars_until_move(
    df: pd.DataFrame,
    *,
    pip_size: float = 0.0001,
    move_pips: float = 30.0,
    max_horizon: int = 200,
    ref_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    bidirectional: bool = True,
) -> pd.Series:
    """
    Time-to-event label:
    number of bars until price moves ±N pips from reference price.
    """

    highs = df[high_col].to_numpy(dtype=float)
    lows = df[low_col].to_numpy(dtype=float)
    refs = df[ref_col].to_numpy(dtype=float)

    n = len(df)
    out = pd.Series([pd.NA] * n, dtype="Int64")

    threshold = move_pips * pip_size

    for i in range(n - 1):
        ref = refs[i]
        upper = ref + threshold
        lower = ref - threshold if bidirectional else None

        for k in range(1, max_horizon + 1):
            j = i + k
            if j >= n:
                break

            if highs[j] >= upper:
                out[i] = k
                break

            if bidirectional and lows[j] <= lower:
                out[i] = k
                break

    return pd.Series(out, index=df.index, name="bars_until_move")
