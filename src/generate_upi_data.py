import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 100_000

df = pd.DataFrame({
    "amount": rng.lognormal(6, 1.2, n).round(2),
    "hour": rng.integers(0, 24, n),
    "account_age_days": rng.integers(1, 2000, n),
    "txn_count_24h": rng.poisson(3, n),
    "new_receiver": rng.integers(0, 2, n),
    "device_changed": rng.binomial(1, 0.05, n),
    "location_mismatch": rng.binomial(1, 0.08, n),
})

night = (df["hour"] < 5).astype(int)

score = (
    -6
    + 0.0002 * df["amount"]
    + 1.5 * df["new_receiver"]
    + 2.0 * df["device_changed"]
    + 1.8 * df["location_mismatch"]
    + 0.15 * df["txn_count_24h"]
    + 0.8 * night
    - 0.0008 * df["account_age_days"]
)

prob = 1 / (1 + np.exp(-score))
df["is_fraud"] = rng.binomial(1, prob)

df.to_csv("data/upi_synthetic.csv", index=False)
print("Saved:", df.shape)
print("Fraud %:", df["is_fraud"].mean() * 100)