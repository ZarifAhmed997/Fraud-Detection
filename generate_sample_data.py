"""
generate_sample_data.py
Generates a realistic synthetic transactions CSV for testing the anomaly detector.
"""
import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

NUM_TRANSACTIONS = 100000
OUTPUT_FILE = "data/transactions.csv"

MERCHANTS = [
    "Amazon", "Tesco", "Sainsbury's", "ASOS", "Argos",
    "Costa Coffee", "Pret A Manger", "Netflix", "Spotify",
    "Shell", "BP", "Marks & Spencer", "John Lewis", "Currys",
]
CATEGORIES = ["retail", "groceries", "dining", "entertainment", "transport", "utilities"]
MERCHANT_CATEGORY = {
    "Amazon": "retail", "Tesco": "groceries", "Sainsbury's": "groceries",
    "ASOS": "retail", "Argos": "retail", "Costa Coffee": "dining",
    "Pret A Manger": "dining", "Netflix": "entertainment", "Spotify": "entertainment",
    "Shell": "transport", "BP": "transport", "Marks & Spencer": "groceries",
    "John Lewis": "retail", "Currys": "retail",
}

start_date = datetime(2024, 1, 1)


def normal_tx(i, date):
    merchant = random.choice(MERCHANTS)
    amount = round(random.expovariate(1 / 45) + 2, 2)  # typical spend £2–£150
    amount = min(amount, 300)
    hour = random.choices(range(24), weights=[
        1,1,1,1,1,1,2,5,8,9,9,9,9,9,8,8,7,7,6,5,4,3,2,1
    ])[0]
    tx_time = date + timedelta(hours=hour, minutes=random.randint(0, 59))
    return {
        "transaction_id": f"TXN{i:05d}",
        "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": amount,
        "merchant": merchant,
        "category": MERCHANT_CATEGORY[merchant],
        "customer_id": f"CUST{random.randint(1, 200):04d}",
        "is_anomaly": 0,
    }


def anomaly_tx(i, date):
    """Three anomaly types: large amount, odd hour, high velocity."""
    type = random.randint(0, 2)
    if type == 0:  # huge amount
        merchant = random.choice(MERCHANTS)
        amount = round(random.uniform(1500, 9500), 2)
        hour = random.randint(8, 20)
    elif type == 1:  # suspicious late-night
        merchant = random.choice(MERCHANTS)
        amount = round(random.uniform(200, 800), 2)
        hour = random.randint(1, 4)
    else:  # tiny round-number test charge
        merchant = random.choice(MERCHANTS)
        amount = round(random.choice([0.01, 0.10, 1.00]), 2)
        hour = random.randint(0, 23)

    tx_time = date + timedelta(hours=hour, minutes=random.randint(0, 59))
    return {
        "transaction_id": f"TXN{i:05d}",
        "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": amount,
        "merchant": merchant if type != 2 else random.choice(MERCHANTS),
        "category": MERCHANT_CATEGORY[merchant],
        "customer_id": f"CUST{random.randint(1, 200):04d}",
        "is_anomaly": 1,
    }


rows = []
for i in range(1, NUM_TRANSACTIONS + 1):
    date = start_date + timedelta(days=random.randint(0, 364))
    if random.random() < 0.07:   # ~7% anomalies
        rows.append(anomaly_tx(i, date))
    else:
        rows.append(normal_tx(i, date))

rows.sort(key=lambda r: r["timestamp"])

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} transactions → {OUTPUT_FILE}")
print(f"  Anomalies: {sum(r['is_anomaly'] for r in rows)}")
print(f"  Normal:    {sum(1 - r['is_anomaly'] for r in rows)}")
