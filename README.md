# Transaction Anomaly Detector

A deep learning command-line tool for detecting anomalous financial transactions in CSV data, using a **TensorFlow Autoencoder** with a full feature engineering pipeline built on raw transaction fields.

---

## Results on Sample Data (1,000 transactions, 7.6% true anomaly rate)

| Method         | ROC-AUC | Precision | Recall | F1    |
|----------------|---------|-----------|--------|-------|
| TF Autoencoder | 0.986   | 0.840     | 0.553  | 0.667 |

---

## Features

- **TensorFlow Autoencoder** — trained only on normal transactions; flags anything it can't reconstruct well
- **Feature engineering** — extracts time-based, velocity, and relative-spend features automatically from raw columns
- **Ground-truth evaluation** — prints precision, recall, F1, and ROC-AUC if your CSV includes an `is_anomaly` column
- **CSV output** — appends anomaly labels and reconstruction error scores to your original data

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

# 2. Run the detector
python anomaly_detector_tf.py --input transactions.csv

# 3. Tune sensitivity
python anomaly_detector_tf.py --input transactions.csv --contamination 0.03

# 4. Train longer and save to a specific file
python anomaly_detector_tf.py --input transactions.csv --epochs 100 --output flagged.csv
```

---

## CLI Reference

```
usage: anomaly_detector_tf.py [-h] --input INPUT
                               [--contamination CONTAMINATION]
                               [--epochs EPOCHS]
                               [--batch-size BATCH_SIZE]
                               [--output OUTPUT]

arguments:
  --input,         -i   Path to input CSV file                       (required)
  --contamination, -c   Expected fraction of anomalies, e.g. 0.05   (default: 0.05)
  --epochs,        -e   Max training epochs; early stopping applies  (default: 50)
  --batch-size,    -b   Training batch size                          (default: 32)
  --output,        -o   Output CSV path          (default: <input>_tf_results.csv)
```

---

## Input CSV Format

| Column           | Type     | Description                                |
|------------------|----------|--------------------------------------------|
| `transaction_id` | string   | Unique transaction identifier              |
| `timestamp`      | datetime | Transaction datetime `YYYY-MM-DD HH:MM:SS` |
| `amount`         | float    | Transaction amount                         |
| `merchant`       | string   | Merchant name                              |
| `customer_id`    | string   | Customer identifier                        |

Optional:

| Column       | Type | Description                                                                                   |
|--------------|------|-----------------------------------------------------------------------------------------------|
| `category`   | str  | Merchant category (e.g. `retail`, `dining`)                                                   |
| `is_anomaly` | int  | Ground-truth label (`0` = normal, `1` = anomaly). If present, evaluation metrics are printed. |

A small example file is included at [`sample_data/sample_transactions.csv`](sample_data/sample_transactions.csv).

---

## Output CSV

The input data is preserved and the following columns are appended:

| Column                | Description                                      |
|-----------------------|--------------------------------------------------|
| `autoencoder_anomaly` | `1` = flagged as anomaly, `0` = normal           |
| `autoencoder_score`   | Reconstruction MSE — higher means more anomalous |

---

## How It Works

### Feature Engineering

Raw transaction columns are transformed into six numeric features before training:

| Feature                 | Description                                             |
|-------------------------|---------------------------------------------------------|
| `log_amount`            | `log(1 + amount)` — compresses the heavy right tail     |
| `hour`                  | Hour of day (0-23)                                      |
| `is_weekend`            | 1 if Saturday or Sunday                                 |
| `is_night`              | 1 if hour < 6 or hour >= 23                             |
| `daily_tx_count`        | Number of transactions by this customer on the same day |
| `amount_vs_cust_median` | Amount divided by this customer's own median spend      |

All features are standardised with `StandardScaler` before being passed to the model.

### Autoencoder Architecture

```
Input (6 features)
  -> Dense 16  (ReLU)
  -> Dense 8   (ReLU)
  -> Dense 4   (ReLU)  <- bottleneck
  -> Dense 8   (ReLU)
  -> Dense 16  (ReLU)
  -> Dense 6   (Linear) <- reconstruction
```

The model is trained **only on normal transactions**, minimising the mean squared error (MSE) between input and reconstruction. When it encounters an anomalous transaction at inference time, it reconstructs it poorly — producing a high MSE score — because it never learned those patterns during training.

The top `contamination` fraction by reconstruction error is flagged as anomalies.

If no ground-truth labels are available, the model trains unsupervised on all transactions. Normal transactions dominate the dataset, so the model still learns primarily normal patterns.

---

## Project Structure

```
transaction-anomaly-detector/
├── anomaly_detector_tf.py     # TensorFlow autoencoder detector
├── generate_sample_data.py    # Synthetic data generator for testing
├── requirements.txt
├── .gitignore
├── README.md
└── sample_data/
    └── sample_transactions.csv
```

---

## Dependencies

| Package      | Purpose                                  |
|--------------|------------------------------------------|
| tensorflow   | Autoencoder model training and inference |
| pandas       | Data loading and feature engineering     |
| numpy        | Numerical operations                     |
| scikit-learn | StandardScaler, evaluation metrics       |

---

## Tuning Tips

- **`--contamination`** is the most important parameter. Set it to your best estimate of the true fraud rate in your data. Lower = stricter flagging.
- **`--epochs`** defaults to 50 with early stopping. If val_loss is still dropping at epoch 50, try 100-200.
- If you have labelled data (`is_anomaly` column), the script prints ROC-AUC automatically — use this to compare contamination values.
- For production use, save the trained model with `model.save()` and load it with `tf.keras.models.load_model()` to score new batches without retraining.

---

## Extending the Project

- Add more features: merchant category encoding, device ID, geolocation delta
- Add a `--plot` flag to generate matplotlib reconstruction error histograms
- Build a FastAPI wrapper to serve predictions over HTTP
- Schedule batch runs with cron and email alerts on flagged transactions

---

## Licence

MIT