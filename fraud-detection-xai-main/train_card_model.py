"""
Train a card-fraud model that matches the features the SecurePay app sends.

Run from the project root (C:\\fraud-detection-xai):
    python train_card_model.py

Output: models/xgb_card.pkl  (dict with 'model' and 'features', which api/main.py understands)

NOTE: the training data is synthetic (generated below), so this is for the demo.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURES = ["amount", "amount_vs_avg", "txn_count_1h", "minutes_since_last",
            "distance_from_last_km", "travel_speed_kmh", "merchant_risk",
            "is_online", "is_foreign"]
RISK_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]  # same values as CATEGORIES in app.py

rng = np.random.default_rng(42)


def make_rows(n, fraud):
    if fraud:
        amount = rng.lognormal(8.6, 0.9, n)                  # typically larger
        amount_vs_avg = rng.lognormal(1.6, 0.6, n)           # much above normal spend
        txn_count_1h = rng.poisson(3.0, n)
        minutes = rng.lognormal(2.5, 1.0, n).clip(1, 10080)  # soon after last txn
        dist = np.where(rng.random(n) < 0.65, rng.uniform(300, 2000, n), rng.uniform(0, 50, n))
        risk = rng.choice(RISK_LEVELS, n, p=[.03, .03, .04, .10, .12, .2, .24, .24])
        online = (rng.random(n) < 0.7).astype(int)
        foreign = (rng.random(n) < 0.35).astype(int)
    else:
        amount = rng.lognormal(7.2, 1.0, n)
        amount_vs_avg = rng.lognormal(0.0, 0.5, n)
        txn_count_1h = rng.poisson(0.25, n)
        minutes = rng.lognormal(6.5, 1.2, n).clip(1, 10080)
        dist = np.where(rng.random(n) < 0.8, 0.0, rng.uniform(0, 1500, n))
        risk = rng.choice(RISK_LEVELS, n, p=[.25, .2, .15, .1, .1, .1, .06, .04])
        online = (rng.random(n) < 0.3).astype(int)
        foreign = (rng.random(n) < 0.03).astype(int)

    speed = np.minimum(dist / (minutes / 60), 3000.0)
    return pd.DataFrame({
        "amount": amount.round(2),
        "amount_vs_avg": amount_vs_avg.round(3),
        "txn_count_1h": txn_count_1h,
        "minutes_since_last": minutes.round(1),
        "distance_from_last_km": dist.round(1),
        "travel_speed_kmh": speed.round(1),
        "merchant_risk": risk,
        "is_online": online,
        "is_foreign": foreign,
        "is_fraud": int(fraud),
    })


df = pd.concat([make_rows(18000, False), make_rows(2000, True)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X, y = df[FEATURES], df["is_fraud"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = XGBClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.08,
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=3, eval_metric="logloss", random_state=42,
)
model.fit(X_tr, y_tr)

proba = model.predict_proba(X_te)[:, 1]
print("ROC-AUC:", round(roc_auc_score(y_te, proba), 4))
print(classification_report(y_te, (proba >= 0.5).astype(int), digits=3))

out = Path("models")
out.mkdir(exist_ok=True)
joblib.dump({"model": model, "features": FEATURES}, out / "xgb_card.pkl")
print("Saved ->", out / "xgb_card.pkl")