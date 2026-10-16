from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
model = joblib.load(BASE / "models" / "xgb_upi.pkl")
scaler = joblib.load(BASE / "models" / "scaler_upi.pkl")
explainer = shap.TreeExplainer(model)

COLS = ["amount", "hour", "account_age_days", "txn_count_24h",
        "new_receiver", "device_changed", "location_mismatch"]
NUM_COLS = ["amount", "account_age_days", "txn_count_24h"]


class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Amount in Rs")
    hour: int = Field(..., ge=0, le=23)
    account_age_days: int = Field(..., ge=1)
    txn_count_24h: int = Field(..., ge=0)
    new_receiver: int = Field(..., ge=0, le=1)
    device_changed: int = Field(..., ge=0, le=1)
    location_mismatch: int = Field(..., ge=0, le=1)


app = FastAPI(title="UPI Fraud Detection API", version="1.0")


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
    contributions = sorted(
        zip(COLS, raw.iloc[0].tolist(), sv.values[0].tolist()),
        key=lambda x: abs(x[2]),
        reverse=True,
    )[:3]

    return {
        "fraud_probability": round(prob, 4),
        "decision": "FRAUD" if prob >= threshold else "NORMAL",
        "threshold": threshold,
        "top_reasons": [
            {
                "feature": f,
                "value": v,
                "impact": round(i, 3),
                "pushes_toward": "fraud" if i > 0 else "normal",
            }
            for f, v, i in contributions
        ],
    }