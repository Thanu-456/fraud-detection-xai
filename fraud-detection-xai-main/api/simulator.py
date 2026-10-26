import csv
import os
import random
import time
from datetime import datetime

import requests

URL = "http://127.0.0.1:8000/predict"
LOG = "data/live_log.csv"
FIELDS = ["time", "amount", "hour", "account_age_days", "txn_count_24h",
          "new_receiver", "device_changed", "location_mismatch",
          "fraud_probability", "decision", "top_reason"]


def make_txn():
    if random.random() < 0.15:  # suspicious
        return {
            "amount": round(random.uniform(15000, 90000), 2),
            "hour": random.choice([0, 1, 2, 3, 4]),
            "account_age_days": random.randint(1, 60),
            "txn_count_24h": random.randint(8, 20),
            "new_receiver": 1,
            "device_changed": random.choice([0, 1, 1]),
            "location_mismatch": random.choice([0, 1, 1]),
        }
    return {  # normal
        "amount": round(random.uniform(50, 3000), 2),
        "hour": random.randint(8, 22),
        "account_age_days": random.randint(100, 1900),
        "txn_count_24h": random.randint(0, 5),
        "new_receiver": random.choice([0, 0, 1]),
        "device_changed": 0,
        "location_mismatch": 0,
    }


os.makedirs("data", exist_ok=True)
new_file = not os.path.exists(LOG)

with open(LOG, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    print("Simulator started. Stop cheyadaniki Ctrl + C nokkandi.")
    count = 0
    while True:
        txn = make_txn()
        try:
            r = requests.post(URL, json=txn, timeout=5).json()
        except Exception as e:
            print("API error:", e)
            time.sleep(2)
            continue

        count += 1
        row = {"time": datetime.now().strftime("%H:%M:%S"), **txn,
               "fraud_probability": r["fraud_probability"],
               "decision": r["decision"],
               "top_reason": r["top_reasons"][0]["feature"]}
        writer.writerow(row)
        f.flush()

        print(f"#{count:03d} Rs {txn['amount']:>9,.2f} | "
              f"prob {r['fraud_probability']:.2f} | {r['decision']:<6} | "
              f"reason: {row['top_reason']}")
        time.sleep(1)