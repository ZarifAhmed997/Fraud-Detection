"""
anomaly_detector.py
Transaction anomaly detection using three ML/statistical methods:
  - Isolation Forest  (scikit-learn)
  - Z-Score           (scipy / numpy)
  - IQR               (numpy)

Usage:
    python anomaly_detector.py --input transactions.csv
    python anomaly_detector.py --input transactions.csv --method all --contamination 0.05
    python anomaly_detector.py --help
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats


# ── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build numeric features from raw transaction columns.
    Works with: transaction_id, timestamp, amount, merchant,
                category, customer_id  (is_anomaly optional).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time-based features
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] < 6) | (df["hour"] >= 23)).astype(int)

    # Customer velocity: how many transactions in the same day?
    df["date"] = df["timestamp"].dt.date
    vel = (
        df.groupby(["customer_id", "date"])["transaction_id"]
        .transform("count")
    )
    df["daily_tx_count"] = vel

    # Amount relative to customer's own history
    cust_median = df.groupby("customer_id")["amount"].transform("median")
    df["amount_vs_cust_median"] = df["amount"] / (cust_median + 1e-9)

    # Log-amount (compress heavy tail)
    df["log_amount"] = np.log1p(df["amount"])

    return df


FEATURE_COLS = [
    "log_amount",
    "hour",
    "is_weekend",
    "is_night",
    "daily_tx_count",
    "amount_vs_cust_median",
]


# ── Detection methods ─────────────────────────────────────────────────────────

def detect_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolation Forest: fits an ensemble of random trees.
    Returns (labels, scores) where label=1 means anomaly.
    Scores are negated so higher = more anomalous.
    """
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    raw_labels = clf.fit_predict(X)          # -1 anomaly, 1 normal
    labels = (raw_labels == -1).astype(int)
    scores = -clf.decision_function(X)       # higher = more anomalous
    return labels, scores


def detect_zscore(
    X: np.ndarray,
    contamination: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-Score: flag points where any feature is far from its mean.
    Uses the max absolute z-score across features as the anomaly score.
    """
    z = np.abs(stats.zscore(X, axis=0))
    scores = z.max(axis=1)
    threshold = np.percentile(scores, 100 * (1 - contamination))
    labels = (scores >= threshold).astype(int)
    return labels, scores


def detect_iqr(
    X: np.ndarray,
    contamination: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    IQR: flag points that lie beyond Q1 - k*IQR or Q3 + k*IQR.
    The outlier score is the max normalised distance past a fence.
    """
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1 + 1e-9
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Distance past each fence, normalised by IQR
    below = np.maximum(0, (lower - X) / iqr)
    above = np.maximum(0, (X - upper) / iqr)
    scores = np.maximum(below, above).max(axis=1)

    threshold = np.percentile(scores, 100 * (1 - contamination))
    labels = (scores >= threshold).astype(int)
    return labels, scores


# ── Report helpers ────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, method: str) -> None:
    col = f"{method}_anomaly"
    score_col = f"{method}_score"
    n_anomalies = df[col].sum()
    pct = n_anomalies / len(df) * 100

    print(f"\n{'─'*55}")
    print(f"  {method.upper().replace('_', ' ')} RESULTS")
    print(f"{'─'*55}")
    print(f"  Total transactions : {len(df):,}")
    print(f"  Flagged anomalies  : {n_anomalies:,}  ({pct:.1f}%)")

    if "is_anomaly" in df.columns:
        from sklearn.metrics import classification_report, roc_auc_score
        print("\n  Classification report (against ground-truth labels):")
        print(classification_report(df["is_anomaly"], df[col],
                                    target_names=["Normal", "Anomaly"],
                                    digits=3))
        try:
            auc = roc_auc_score(df["is_anomaly"], df[score_col])
            print(f"  ROC-AUC : {auc:.4f}")
        except Exception:
            pass

    print(f"\n  Top 5 most suspicious transactions:")
    top = (
        df.nlargest(5, score_col)[
            ["transaction_id", "timestamp", "amount", "merchant",
             "customer_id", score_col, col]
        ]
        .rename(columns={score_col: "score", col: "flagged"})
    )
    print(top.to_string(index=False))


def save_results(df: pd.DataFrame, output_path: str) -> None:
    cols = [c for c in df.columns if not c.startswith("__")]
    df[cols].to_csv(output_path, index=False)
    print(f"\n  Results saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transaction anomaly detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--method", "-m",
        choices=["isolation_forest", "zscore", "iqr", "all"],
        default="all",
        help="Detection method to use",
    )
    parser.add_argument(
        "--contamination", "-c",
        type=float,
        default=0.05,
        help="Expected fraction of anomalies (0.01–0.30)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: <input>_results.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load & validate ───────────────────────────────────────────────────────
    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Error: file not found: {args.input}")

    print(f"\nLoading {args.input} …")
    df = pd.read_csv(args.input)
    required = {"transaction_id", "timestamp", "amount", "merchant", "customer_id"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: CSV is missing columns: {missing}")

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ── Feature engineering ───────────────────────────────────────────────────
    print("\nEngineering features …")
    df = engineer_features(df)
    X_raw = df[FEATURE_COLS].fillna(0).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ── Run selected methods ──────────────────────────────────────────────────
    methods = (
        ["isolation_forest", "zscore", "iqr"]
        if args.method == "all"
        else [args.method]
    )

    for method in methods:
        print(f"\nRunning {method} (contamination={args.contamination}) …")
        if method == "isolation_forest":
            labels, scores = detect_isolation_forest(X, args.contamination)
        elif method == "zscore":
            labels, scores = detect_zscore(X, args.contamination)
        elif method == "iqr":
            labels, scores = detect_iqr(X, args.contamination)

        df[f"{method}_anomaly"] = labels
        df[f"{method}_score"]   = scores.round(6)

        print_summary(df, method)

    # ── Ensemble vote (if all methods run) ───────────────────────────────────
    if args.method == "all":
        vote_cols = [f"{m}_anomaly" for m in methods]
        df["ensemble_votes"]   = df[vote_cols].sum(axis=1)
        df["ensemble_anomaly"] = (df["ensemble_votes"] >= 2).astype(int)

        n_ens = df["ensemble_anomaly"].sum()
        print(f"\n{'─'*55}")
        print(f"  ENSEMBLE (majority vote ≥ 2/3 methods)")
        print(f"{'─'*55}")
        print(f"  Flagged by ensemble : {n_ens:,}  ({n_ens/len(df)*100:.1f}%)")
        if "is_anomaly" in df.columns:
            from sklearn.metrics import classification_report
            print(classification_report(
                df["is_anomaly"], df["ensemble_anomaly"],
                target_names=["Normal", "Anomaly"], digits=3
            ))

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = args.output or str(path.with_name(path.stem + "_results.csv"))
    # Drop intermediate columns before saving
    drop_cols = ["date"]
    df_out = df.drop(columns=[c for c in drop_cols if c in df.columns])
    save_results(df_out, output_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
