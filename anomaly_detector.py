"""
anomaly_detector_tf.py
----------------------
Transaction anomaly detection using a TensorFlow Autoencoder.

How it works:
  The autoencoder is trained ONLY on normal transactions. It learns to
  compress and reconstruct them accurately. When it sees an anomalous
  transaction, it reconstructs it poorly — giving a high reconstruction
  error (MSE), which becomes the anomaly score.

  Architecture:
    Input (6 features)
      -> Encoder: 6 -> 16 -> 8 -> 4  (bottleneck)
      -> Decoder: 4 -> 8 -> 16 -> 6  (reconstruction)
    Loss: Mean Squared Error between input and reconstruction

Usage:
    python anomaly_detector_tf.py --input transactions.csv
    python anomaly_detector_tf.py --input transactions.csv --contamination 0.05
    python anomaly_detector_tf.py --input transactions.csv --epochs 100 --output flagged.csv
    python anomaly_detector_tf.py --help

Required CSV columns: transaction_id, timestamp, amount, merchant, customer_id
Optional CSV column:  is_anomaly (0/1) -- enables automatic evaluation metrics
"""

import argparse
import os
import sys
from pathlib import Path

# Suppress TensorFlow info/warning logs -- only show errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler


# -- Feature engineering (same pipeline as the original) ----------------------

FEATURE_COLS = [
    "log_amount",
    "hour",
    "is_weekend",
    "is_night",
    "daily_tx_count",
    "amount_vs_cust_median",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric features from raw transaction columns."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] < 6) | (df["hour"] >= 23)).astype(int)

    df["date"] = df["timestamp"].dt.date
    df["daily_tx_count"] = (
        df.groupby(["customer_id", "date"])["transaction_id"].transform("count")
    )

    cust_median = df.groupby("customer_id")["amount"].transform("median")
    df["amount_vs_cust_median"] = df["amount"] / (cust_median + 1e-9)
    df["log_amount"] = np.log1p(df["amount"])

    return df


# -- Autoencoder model ---------------------------------------------------------

def build_autoencoder(input_dim: int) -> tf.keras.Model:
    """
    Symmetric encoder-decoder network.

    Encoder compresses the input down to a 4-dimensional bottleneck,
    forcing the model to learn only the most important patterns in
    normal transactions. The decoder then reconstructs the original
    input from that bottleneck representation.

    Anomalies have patterns the model never learned -> poor reconstruction
    -> high MSE -> high anomaly score.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="input")

    # Encoder: progressively compress
    x = tf.keras.layers.Dense(16, activation="relu", name="enc_1")(inputs)
    x = tf.keras.layers.Dense(8,  activation="relu", name="enc_2")(x)
    encoded = tf.keras.layers.Dense(4, activation="relu", name="bottleneck")(x)

    # Decoder: progressively reconstruct
    x = tf.keras.layers.Dense(8,  activation="relu",   name="dec_1")(encoded)
    x = tf.keras.layers.Dense(16, activation="relu",   name="dec_2")(x)
    decoded = tf.keras.layers.Dense(input_dim, activation="linear", name="output")(x)

    model = tf.keras.Model(inputs, decoded, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


# -- Detection -----------------------------------------------------------------

def detect_autoencoder(
    X: np.ndarray,
    contamination: float = 0.05,
    epochs: int = 50,
    batch_size: int = 32,
    has_labels: bool = False,
    y_true: np.ndarray = None,
):
    """
    Train the autoencoder, then score every transaction by reconstruction error.

    If ground-truth labels are available (is_anomaly column in CSV), the model
    trains only on normal transactions -- this gives the best results and mirrors
    real production usage where you train on verified clean history.

    Without labels, the model trains unsupervised on everything. It still works
    because normal transactions are the majority and dominate the loss.

    Returns: (labels, scores, trained_model)
      labels: np.ndarray of 0/1  (1 = anomaly)
      scores: np.ndarray of floats (reconstruction MSE, higher = more anomalous)
    """
    if has_labels and y_true is not None:
        X_train = X[y_true == 0]
        print(f"  Supervised mode: training on {len(X_train):,} normal transactions")
    else:
        X_train = X
        print(f"  Unsupervised mode: training on all {len(X_train):,} transactions")

    model = build_autoencoder(input_dim=X.shape[1])

    print(f"\n  Model architecture:")
    model.summary(print_fn=lambda s: print(f"    {s}"))

    print(f"\n  Training for up to {epochs} epochs (early stopping enabled) ...")
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=0,
            )
        ],
        verbose=0,
    )

    epochs_ran  = len(history.history["val_loss"])
    final_loss  = history.history["val_loss"][-1]
    print(f"  Stopped at epoch {epochs_ran}  |  final val_loss: {final_loss:.6f}")

    # Anomaly score = per-sample mean squared reconstruction error
    reconstructions = model.predict(X, verbose=0)
    scores = np.mean(np.square(X - reconstructions), axis=1)

    # Flag the top `contamination` fraction as anomalies
    threshold = np.percentile(scores, 100 * (1 - contamination))
    labels = (scores >= threshold).astype(int)

    return labels, scores, model


# -- Report --------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    n_anomalies = df["autoencoder_anomaly"].sum()
    pct = n_anomalies / len(df) * 100

    print(f"\n{'─'*55}")
    print(f"  AUTOENCODER RESULTS")
    print(f"{'─'*55}")
    print(f"  Total transactions : {len(df):,}")
    print(f"  Flagged anomalies  : {n_anomalies:,}  ({pct:.1f}%)")

    if "is_anomaly" in df.columns:
        print("\n  Classification report (against ground-truth labels):")
        print(classification_report(
            df["is_anomaly"],
            df["autoencoder_anomaly"],
            target_names=["Normal", "Anomaly"],
            digits=3,
        ))
        try:
            auc = roc_auc_score(df["is_anomaly"], df["autoencoder_score"])
            print(f"  ROC-AUC : {auc:.4f}")
        except Exception:
            pass

    print("\n  Top 10 most suspicious transactions:")
    top = (
        df.nlargest(10, "autoencoder_score")[
            ["transaction_id", "timestamp", "amount", "merchant",
             "customer_id", "autoencoder_score", "autoencoder_anomaly"]
        ]
        .rename(columns={
            "autoencoder_score":   "recon_error",
            "autoencoder_anomaly": "flagged",
        })
    )
    print(top.to_string(index=False))


def save_results(df: pd.DataFrame, output_path: str) -> None:
    drop_cols = ["date"]
    df.drop(columns=[c for c in drop_cols if c in df.columns]).to_csv(
        output_path, index=False
    )
    print(f"\n  Results saved -> {output_path}")


# -- CLI -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorFlow autoencoder transaction anomaly detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to input CSV file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: <input>_tf_results.csv)")
    parser.add_argument("--contamination", "-c", type=float, default=0.05,
                        help="Expected fraction of anomalies (0.01-0.30)")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="Max training epochs (early stopping may halt sooner)")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Training batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    # -- Load & validate -------------------------------------------------------
    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Error: file not found: {args.input}")

    print(f"\nLoading {args.input} ...")
    df = pd.read_csv(args.input)

    required = {"transaction_id", "timestamp", "amount", "merchant", "customer_id"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: CSV is missing columns: {missing}")

    has_labels = "is_anomaly" in df.columns
    print(f"  Loaded {len(df):,} rows  |  ground-truth labels: {'yes' if has_labels else 'no'}")

    # -- Feature engineering ---------------------------------------------------
    print("\nEngineering features ...")
    df = engineer_features(df)
    X_raw = df[FEATURE_COLS].fillna(0).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    y_true = df["is_anomaly"].values if has_labels else None

    # -- Train & detect --------------------------------------------------------
    print(f"\nBuilding autoencoder  |  contamination={args.contamination}")
    labels, scores, _ = detect_autoencoder(
        X,
        contamination=args.contamination,
        epochs=args.epochs,
        batch_size=args.batch_size,
        has_labels=has_labels,
        y_true=y_true,
    )

    df["autoencoder_anomaly"] = labels
    df["autoencoder_score"]   = scores.round(6)

    # -- Report & save ---------------------------------------------------------
    print_summary(df)

    output_path = args.output or str(path.with_name(path.stem + "_tf_results.csv"))
    save_results(df, output_path)
    print("\nDone.")


if __name__ == "__main__":
    main()