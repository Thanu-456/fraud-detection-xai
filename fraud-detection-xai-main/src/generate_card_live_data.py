import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 120_000

amount = rng.lognormal(7, 1, n).round(2)
ratio = rng.lognormal(0, 0.6, n)
big = rng.random(n) < 0.03
ratio = np.where(big, ratio * rng.uniform(3, 15, n), ratio)
minutes = np.clip(rng.exponential(600, n), 1, 10080)
far = rng.random(n) < 0.05
distance = np.where(far, rng.uniform(300, 2000, n), rng.exponential(15, n))
speed = np.clip(distance / (minutes / 60), 0, 3000)

df = pd.DataFrame({
    "amount": amount,
    "amount_vs_avg": ratio.round(3),
    "txn_count_1h": rng.poisson(0.7, n),
    "minutes_since_last": minutes.round(1),
    "distance_from_last_km": distance.round(1),
    "travel_speed_kmh": speed.round(1),
    "merchant_risk": rng.choice([0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], n,
                                p=[.30, .15, .15, .08, .07, .10, .10, .05]),
    "is_online": rng.binomial(1, 0.3, n),
    "is_foreign": rng.binomial(1, 0.04, n),
})

score = (
    -7.5
    + 0.7 * np.clip(df["amount_vs_avg"] - 1, 0, 10)
    + 0.00002 * df["amount"]
    + 0.6 * df["txn_count_1h"]
    + 2.0 * (df["travel_speed_kmh"] > 500)
    + 0.8 * (df["minutes_since_last"] < 5)
    + 1.5 * df["merchant_risk"]
    + 1.0 * df["is_online"]
    + 1.5 * df["is_foreign"]
)
df["is_fraud"] = rng.binomial(1, 1 / (1 + np.exp(-score)))

os.makedirs("data", exist_ok=True)
df.to_csv("data/card_live_synthetic.csv", index=False)
print(df.shape, "Fraud %:", round(df["is_fraud"].mean() * 100, 2))