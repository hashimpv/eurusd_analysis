from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from label_30pip_horizon import hit_within_next_bars


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    eps = 1e-12
    rng = (h - l).clip(lower=eps)

    x = pd.DataFrame(index=df.index)

    x["ret1"] = c.pct_change()
    x["ret3"] = c.pct_change(3)
    x["ret5"] = c.pct_change(5)

    x["range_rel"] = (h - l) / c.abs().clip(lower=eps)
    x["body_rel"] = (c - o) / c.abs().clip(lower=eps)
    x["close_in_range"] = (c - l) / rng

    pc = c.pct_change()
    x["vol10"] = pc.shift(1).rolling(10).std()
    x["vol36"] = pc.shift(1).rolling(36).std()

    x["range_ma12"] = ((h - l) / c.abs().clip(lower=eps)
                       ).shift(1).rolling(12).mean()

    return x


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    required = ["open", "high", "low", "close"]
    if not all(r in cols for r in required):
        raise ValueError("Missing OHLC columns")

    return pd.DataFrame(
        {
            "open": df[cols["open"]],
            "high": df[cols["high"]],
            "low": df[cols["low"]],
            "close": df[cols["close"]],
        }
    )


def train_test_split_time(n: int, train_frac: float, val_frac: float):
    tr = int(n * train_frac)
    va = int(n * (train_frac + val_frac))
    idx = np.arange(n)

    return idx[:tr], idx[tr:va], idx[va:]


def evaluate_threshold(y_true, p, threshold: float):
    pred = (p >= threshold).astype(int)

    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def find_best_threshold(y, p, target_precision: float):
    best_t = None
    best_rate = 0
    best_prec = 0

    for t in np.linspace(0.1, 0.99, 200):
        mask = p >= t
        if mask.sum() < 5:
            continue

        prec = float(y[mask].mean())
        rate = mask.mean()

        if prec >= target_precision:
            return t, rate, prec

        if prec > best_prec:
            best_prec = prec
            best_t = t
            best_rate = rate

    return best_t, best_rate, best_prec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--target-precision", type=float, default=0.9)
    args = parser.parse_args()

    df = load_csv(args.csv)

    y = hit_within_next_bars(df, args.horizon)
    X = build_features(df)

    data = pd.concat([X, y.rename("y")], axis=1).dropna()

    yv = data["y"].astype(int).to_numpy()
    Xv = data.drop(columns=["y"]).to_numpy()

    n = len(data)
    tr, va, te = train_test_split_time(n, 0.7, 0.15)

    X_tr, y_tr = Xv[tr], yv[tr]
    X_va, y_va = Xv[va], yv[va]
    X_te, y_te = Xv[te], yv[te]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_tr, y_tr)

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    print("\n=== BASELINE ===")
    print("Train positive rate:", float(y_tr.mean()))

    print("\n=== VALIDATION ===")
    print(evaluate_threshold(y_va, p_va, 0.5))

    print("\n=== TEST ===")
    print(evaluate_threshold(y_te, p_te, 0.5))

    print("\n=== HIGH PRECISION MODE ===")

    t, rate, prec = find_best_threshold(y_va, p_va, args.target_precision)

    print(f"Best threshold: {t}")
    print(f"Validation precision: {prec:.3f}")
    print(f"Alert rate: {rate:.3f}")

    if t is not None:
        test_metrics = evaluate_threshold(y_te, p_te, t)
        print("\nTest at same threshold:")
        print(test_metrics)


if __name__ == "__main__":
    main()
