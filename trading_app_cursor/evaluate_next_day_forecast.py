from __future__ import annotations
from indicators.next_day_candle import evaluate_next_day_model
from db.db_connect import get_connection

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_from_db() -> pd.DataFrame:
    conn = get_connection()

    df = pd.read_sql(
        """
        SELECT timestamp, open, high, low, close
        FROM market_data
        WHERE symbol = 'EURUSD' AND timeframe = '1D'
        ORDER BY timestamp ASC;
        """,
        conn,
    )

    conn.close()
    return df


def print_metrics(out: dict) -> None:
    print("\n=== Next-Day Model Evaluation ===\n")

    print(f"Total rows: {out['n_total']}")
    print(f"Train: {out['n_train']} | Test: {out['n_test']}\n")

    print(f"Accuracy: {out['test_accuracy_top1']:.3f}")
    print(f"Baseline: {out['baseline_always_predict_train_mode']:.3f}\n")

    print(
        f"Mean P(true class): {out['mean_probability_assigned_to_true_class']:.3f}")
    print(f"Mean max probability: {out['mean_max_predicted_probability']:.3f}")
    print(f"Log loss: {out['multiclass_log_loss']:.3f}")
    print(f"Brier score: {out['multiclass_brier_score']:.3f}\n")

    print(
        f"Accuracy (p ≥ 0.35): {out['test_accuracy_if_maxprob_ge_0.35']:.3f} "
        f"(n={out['n_test_if_maxprob_ge_0.35']})"
    )

    print(
        f"Accuracy (p ≥ 0.50): {out['test_accuracy_if_maxprob_ge_0.50']:.3f} "
        f"(n={out['n_test_if_maxprob_ge_0.50']})"
    )

    print("\nClass distribution (test):")
    print(out["class_distribution_test"])

    print("\nClassification report:")
    print(out["classification_report"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate next-day candle model")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--test-holdout", type=int, default=60)

    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = load_from_db()

    result = evaluate_next_day_model(df, test_holdout=args.test_holdout)

    if not result.get("ok"):
        print("Error:", result.get("error"))
        sys.exit(1)

    print_metrics(result)


if __name__ == "__main__":
    main()
