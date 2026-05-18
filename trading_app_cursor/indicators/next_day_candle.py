from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

PIP = 0.0001


def classify_body(body_pips: float) -> str:
    if np.isnan(body_pips):
        return ""
    if body_pips >= 30:
        return "strong_bull"
    if body_pips <= -30:
        return "strong_bear"
    if body_pips > 0:
        return "bull"
    return "bear"


def candle_type_from_open_close(open_: float, close_: float) -> str:
    return classify_body((close_ - open_) / PIP) or "bear"


def next_day_labels(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"].shift(-1)
    close_ = df["close"].shift(-1)
    body = (close_ - open_) / PIP

    labels = body.map(classify_body)
    labels = labels.replace("", np.nan)

    return pd.Series(labels, index=df.index, name="next_day_type")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    eps = 1e-12
    rng = (h - l).clip(lower=eps)

    x = pd.DataFrame(index=df.index)

    x["body_pips"] = (c - o) / PIP
    x["range_pips"] = (h - l) / PIP
    x["close_in_range"] = (c - l) / rng
    x["bull_sweep_pips"] = (o - l) / PIP
    x["bear_sweep_pips"] = (h - o) / PIP

    x["ret1"] = c.pct_change()
    x["ret2"] = c.pct_change(2)
    x["ret5"] = c.pct_change(5)

    pc = c.pct_change()
    x["vol5"] = pc.shift(1).rolling(5).std()
    x["vol20"] = pc.shift(1).rolling(20).std()

    return x


def _prepare_xy(df: pd.DataFrame, min_rows: int):
    required = {"open", "high", "low", "close"}

    if not required.issubset(set(df.columns.str.lower())):
        return pd.DataFrame(), pd.Series(dtype=object), "Missing columns"

    d = df.copy()

    d.columns = [c.lower() for c in d.columns]

    d = d.sort_values(
        "timestamp" if "timestamp" in d.columns else d.index).reset_index(drop=True)

    y = next_day_labels(d)
    X = build_features(d)

    data = pd.concat([X, y], axis=1).dropna()

    if len(data) < min_rows:
        return pd.DataFrame(), pd.Series(dtype=object), "Not enough rows"

    return data.drop(columns=["next_day_type"]), data["next_day_type"].astype(str), None


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_depth=5,
                    max_iter=120,
                    min_samples_leaf=20,
                    l2_regularization=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_next_day_model(df: pd.DataFrame, min_rows: int = 120, test_size: int = 60) -> dict:
    X, y, err = _prepare_xy(df, min_rows)
    if err:
        return {"ok": False, "error": err}

    n = len(X)
    if n <= test_size + 50:
        return {"ok": False, "error": "Insufficient data"}

    split = n - test_size

    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = make_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)

    baseline = (y_test == y_train.mode().iloc[0]).mean()

    classes = list(model.classes_)
    idx = np.array([classes.index(v) for v in y_test])

    p_true = proba[np.arange(len(y_test)), idx]

    return {
        "ok": True,
        "test_accuracy_top1": float(acc),
        "baseline": float(baseline),
        "mean_true_prob": float(np.mean(p_true)),
        "log_loss": float(log_loss(y_test, proba)),
        "report": classification_report(y_test, preds, zero_division=0),
    }


def fit_predict_next_day(df: pd.DataFrame) -> dict:
    X, y, err = _prepare_xy(df, 120)
    if err:
        return {"ok": False, "error": err}

    model = make_model()
    model.fit(X, y)

    last_X = build_features(df).iloc[[-1]]
    probs = model.predict_proba(last_X)[0]

    classes = model.classes_
    series = pd.Series(probs, index=classes).sort_values(ascending=False)

    return {
        "ok": True,
        "classes": list(classes),
        "probs": series,
        "predicted_class": series.index[0],
        "predicted_top_prob": float(series.iloc[0]),
    }
