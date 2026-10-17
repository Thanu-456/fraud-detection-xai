from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent

# ---- UPI model ----
model = joblib.load(BASE / "models" / "xgb_upi.pkl")
scaler = joblib.load(BASE / "models" / "scaler_upi.pkl")
explainer = shap.TreeExplainer(model)

COLS = ["amount", "hour", "account_age_days", "txn_count_24h",
        "new_receiver", "device_changed", "location_mismatch"]
NUM_COLS = ["amount", "account_age_days", "txn_count_24h"]

# ---- Card model ----
card_model = joblib.load(BASE / "models" / "xgb_card_live.pkl")
card_explainer = shap.TreeExplainer(card_model)
CARD_COLS = ["amount", "amount_vs_avg", "txn_count_1h", "minutes_since_last",
             "distance_from_last_km", "travel_speed_kmh", "merchant_risk",
             "is_online", "is_foreign"]


class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Amount in Rs")
    hour: int = Field(..., ge=0, le=23)
    account_age_days: int = Field(..., ge=1)
    txn_count_24h: int = Field(..., ge=0)
    new_receiver: int = Field(..., ge=0, le=1)
    device_changed: int = Field(..., ge=0, le=1)
    location_mismatch: int = Field(..., ge=0, le=1)


class CardTxn(BaseModel):
    amount: float = Field(..., gt=0)
    amount_vs_avg: float = Field(..., ge=0)
    txn_count_1h: int = Field(..., ge=0)
    minutes_since_last: float = Field(..., ge=0)
    distance_from_last_km: float = Field(..., ge=0)
    travel_speed_kmh: float = Field(..., ge=0)
    merchant_risk: float = Field(..., ge=0, le=1)
    is_online: int = Field(..., ge=0, le=1)
    is_foreign: int = Field(..., ge=0, le=1)


def top_reasons(cols, raw, sv):
    contributions = sorted(
        zip(cols, raw.iloc[0].tolist(), sv.values[0].tolist()),
        key=lambda x: abs(x[2]),
        reverse=True,
    )[:3]
    return [
        {"feature": f, "value": v, "impact": round(i, 3),
         "pushes_toward": "fraud" if i > 0 else "normal"}
        for f, v, i in contributions
    ]


app = FastAPI(title="Fraud Detection API", version="2.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction, threshold: float = 0.5):
    raw = pd.DataFrame([txn.model_dump()])[COLS].astype(float)
    scaled = raw.copy()
    scaled[NUM_COLS] = scaler.transform(raw[NUM_COLS])

    prob = float(model.predict_proba(scaled)[0, 1])
    sv = explainer(scaled)

    return {
        "fraud_probability": round(prob, 4),
        "decision": "FRAUD" if prob >= threshold else "NORMAL",
        "threshold": threshold,
        "top_reasons": top_reasons(COLS, raw, sv),
    }


@app.post("/predict-card")
def predict_card(txn: CardTxn, threshold: float = 0.5):
    raw = pd.DataFrame([txn.model_dump()])[CARD_COLS].astype(float)

    prob = float(card_model.predict_proba(raw)[0, 1])
    sv = card_explainer(raw)

    return {
        "fraud_probability": round(prob, 4),
        "decision": "FRAUD" if prob >= threshold else "NORMAL",
        "threshold": threshold,
        "top_reasons": top_reasons(CARD_COLS, raw, sv),
    }