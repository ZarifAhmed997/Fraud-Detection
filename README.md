Transaction-Anomaly-Detection# Transaction Anomaly Detector

A machine learning command-line tool for detecting anomalous financial transactions in CSV data. Implements three detection methods — **Isolation Forest**, **Z-Score**, and **IQR** — plus a majority-vote ensemble, with a full feature engineering pipeline built on top of raw transaction fields.

---

## Features

- **Three ML/statistical algorithms** with tunable sensitivity
- **Ensemble voting** — flags transactions caught by ≥ 2/3 methods for higher confidence
- **Feature engineering** — extracts time-based, velocity, and relative-spend features automatically from raw columns
- **Ground-truth evaluation** — prints precision, recall, F1, and ROC-AUC if your CSV includes an `is_anomaly` column
- **CSV output** — appends anomaly labels and scores to your original data for downstream analysis

---

## Results on Sample Data (1,000 transactions, 7.6% true anomaly rate)

| Method           | ROC-AUC | Precision | Recall | F1   |
|------------------|---------|-----------|--------|------|
| Isolation Forest | 0.960   | 0.660     | 0.434  | 0.524|
| Z-Score          | 0.953   | 0.429     | 0.592  | 0.497|
| IQR              | 0.886   | 0.348     | 0.421  | 0.381|

> **Isolation Forest** achieves the best ROC-AUC and is recommended as the default method for most use cases.

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/your-username/transaction-anomaly-detector.git
cd transaction-anomaly-detector
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Generate sample data to try the tool immediately
python generate_sample_data.py

# 2. Run all three methods with ensemble voting
python anomaly_detector.py --input transactions.csv

# 3. Use a specific method
python anomaly_detector.py --input transactions.csv --method isolation_forest

# 4. Tune sensitivity (fraction of data expected to be anomalous)
python anomaly_detector.py --input transactions.csv --contamination 0.03

# 5. Specify output path
python anomaly_detector.py --input transactions.csv --output flagged_transactions.csv
```

---

## CLI Reference

```
usage: anomaly_detector.py [-h] --input INPUT
                           [--method {isolation_forest,zscore,iqr,all}]
                           [--contamination CONTAMINATION]
                           [--output OUTPUT]

arguments:
  --input,   -i    Path to input CSV file                        (required)
  --method,  -m    Detection method: isolation_forest | zscore | iqr | all
                                                                 (default: all)
  --contamination, -c
                   Expected fraction of anomalies, e.g. 0.05    (default: 0.05)
  --output,  -o    Output CSV path                               (default: <input>_results.csv)
```

---

## Input CSV Format

Your CSV must contain these columns:

| Column           | Type     | Description                              |
|------------------|----------|------------------------------------------|
| `transaction_id` | string   | Unique transaction identifier            |
| `timestamp`      | datetime | Transaction datetime `YYYY-MM-DD HH:MM:SS` |
| `amount`         | float    | Transaction amount                       |
| `merchant`       | string   | Merchant name                            |
| `customer_id`    | string   | Customer identifier                      |

Optional:

| Column       | Type | Description                                              |
|--------------|------|----------------------------------------------------------|
| `category`   | str  | Merchant category (e.g. `retail`, `dining`)              |
| `is_anomaly` | int  | Ground-truth label (`0` = normal, `1` = anomaly). If present, evaluation metrics are printed automatically. |

A small example file is included at [`sample_data/sample_transactions.csv`](sample_data/sample_transactions.csv).

---

## Output CSV

The input data is preserved and the following columns are appended:

| Column                    | Description                                      |
|---------------------------|--------------------------------------------------|
| `isolation_forest_anomaly`| `1` = flagged, `0` = normal                      |
| `isolation_forest_score`  | Anomaly score (higher = more suspicious)         |
| `zscore_anomaly`          | `1` = flagged, `0` = normal                      |
| `zscore_score`            | Max absolute z-score across features             |
| `iqr_anomaly`             | `1` = flagged, `0` = normal                      |
| `iqr_score`               | Max normalised fence distance across features    |
| `ensemble_votes`          | Number of methods (0–3) that flagged this row    |
| `ensemble_anomaly`        | `1` if flagged by ≥ 2 methods                    |

---

## How It Works

### Feature Engineering

Raw transaction columns are transformed into six numeric features before any model runs:

| Feature                  | Description                                               |
|--------------------------|-----------------------------------------------------------|
| `log_amount`             | `log(1 + amount)` — compresses the heavy right tail       |
| `hour`                   | Hour of day (0–23)                                        |
| `is_weekend`             | 1 if Saturday or Sunday                                   |
| `is_night`               | 1 if hour < 6 or hour ≥ 23                                |
| `daily_tx_count`         | Number of transactions by this customer on the same day   |
| `amount_vs_cust_median`  | Amount divided by this customer's own median spend        |

All features are standardised with `StandardScaler` before being passed to the models.

### Algorithms

**Isolation Forest**
Builds an ensemble of random decision trees. Anomalies are isolated in fewer splits because they're rare and extreme, yielding a high anomaly score. Best overall performance and handles multi-dimensional patterns well.

**Z-Score**
Flags transactions where any feature lies far from the population mean (measured in standard deviations). Fast and interpretable. Assumes roughly normal feature distributions.

**IQR (Interquartile Range)**
Flags transactions that fall beyond `Q1 − 1.5×IQR` or `Q3 + 1.5×IQR` on any feature. More robust than Z-Score for skewed distributions.

**Ensemble**
A transaction is flagged if ≥ 2 of the 3 methods agree. Reduces false positives at the cost of slightly lower recall.

---

## Project Structure

```
transaction-anomaly-detector/
├── anomaly_detector.py        # Main detection script
├── generate_sample_data.py    # Synthetic data generator for testing
├── requirements.txt
├── .gitignore
├── README.md
└── sample_data/
    └── sample_transactions.csv
```

---

## Dependencies

| Package      | Purpose                          |
|--------------|----------------------------------|
| pandas       | Data loading and feature engineering |
| numpy        | Numerical operations             |
| scikit-learn | Isolation Forest, StandardScaler, metrics |
| scipy        | Z-Score calculation              |

---

## Tuning Tips

- **`--contamination`** is the most important parameter. Set it to your best estimate of the true fraud rate in your data. Lower values = stricter flagging.
- **Isolation Forest** is the best default for most datasets. Switch to **Z-Score** if you need fast, explainable results. Use **IQR** for heavily skewed amount distributions.
- If you have labelled data (`is_anomaly` column), the script prints ROC-AUC automatically — use this to compare methods and tune contamination.
- For production use, fit the scaler and model on historical clean data, then apply `.transform()` / `.predict()` to new batches rather than re-fitting on every run.

---

## Extending the Project

Some ideas for taking this further:

- Add more features: merchant category encoding, device ID, geolocation delta
- Swap in `LocalOutlierFactor` or `AutoEncoder` as additional methods
- Add a `--plot` flag to generate matplotlib scatter/histogram charts
- Build a FastAPI wrapper to serve predictions over HTTP
- Schedule batch runs with cron and email alerts on flagged transactions

---

## Licence

MIT
realtime-anomaly-engine/
├── README.md
├── CMakeLists.txt
├── .gitignore

├── src/
│   ├── main.cpp                # Entry point (wiring + config)
│   │
│   ├── engine/
│   │   ├── engine.hpp
│   │   ├── engine.cpp          # Core event-processing loop
│   │
│   ├── ingest/
│   │   ├── parser.hpp
│   │   ├── parser.cpp          # CSV / binary event parsing
│   │   ├── replay.hpp
│   │   └── replay.cpp          # Deterministic replay + rate control
│   │
│   ├── features/
│   │   ├── feature_store.hpp
│   │   ├── feature_store.cpp   # Sliding windows, aggregations
│   │   ├── statistics.hpp
│   │   └── statistics.cpp      # Mean, variance, entropy, velocity
│   │
│   ├── ml/
│   │   ├── inference.hpp
│   │   └── inference.cpp       # Lightweight model inference
│   │
│   ├── concurrency/
│   │   ├── ring_buffer.hpp     # Lock-free queues
│   │   └── thread_pool.hpp
│   │
│   ├── memory/
│   │   ├── object_pool.hpp     # Custom allocator
│   │   └── aligned_alloc.hpp   # Cache-line alignment
│   │
│   ├── utils/
│   │   ├── timestamp.hpp
│   │   ├── config.hpp
│   │   └── logging.hpp
│
├── ml/
│   ├── data/
│   │   └── features.csv        # Exported features from C++
│   │
│   ├── train.py                # Train anomaly model
│   ├── evaluate.py             # Metrics, ROC, precision/recall
│   └── export_model.py         # Export to ONNX / lightweight format
│
├── data/
│   ├── raw/
│   │   └── transactions.csv    # Synthetic or public dataset
│   │
│   └── generated/
│       └── events.bin          # Optional binary replay format
│
├── benchmarks/
│   ├── latency.cpp             # End-to-end latency tests
│   └── throughput.cpp          # Events/sec benchmarks
│
├── tests/
│   ├── test_features.cpp
│   ├── test_engine.cpp
│   └── test_parser.cpp
│
├── scripts/
│   ├── generate_data.py        # Synthetic data generator
│   └── run_benchmarks.sh
│
└── docs/
    ├── architecture.md         # High-level design
    ├── performance.md          # Latency + throughput results
    └── ml_pipeline.md          # Feature + ML explanation

