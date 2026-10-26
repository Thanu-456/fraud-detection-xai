"""
fraud_engine.py - model loading + fraud scoring with SHAP explanations.

Used directly by the Streamlit app:
    from fraud_engine import score
    result = score("upi", {...features...}, 0.5)

Place this file in the project root (C:\\fraud-detection-xai), next to app.py.
"""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent

# Folders searched for model files. Set MODEL_DIR to add your own.
MODEL_DIRS = [ROOT / "models", ROOT / "api", ROOT / "artifacts"]
if os.environ.get("MODEL_DIR"):
    MODEL_DIRS.insert(0, Path(os.environ["MODEL_DIR"]))

# Or point straight at the files:
#   set UPI_MODEL_PATH=C:\fraud-detection-xai\models\upi_model.pkl
#   set CARD_MODEL_PATH=C:\fraud-detection-xai\models\card_model.pkl
ENV_PATHS = {"upi": "UPI_MODEL_PATH", "card": "CARD_MODEL_PATH"}

MODEL_EXTENSIONS = {".joblib", ".pkl", ".json", ".ubj"}
SKIP_WORDS = ("scaler", "explainer", "encoder", "metrics", "report", "feature_list")

DEFAULT_FEATURES = {
    "upi": ["amount", "hour", "account_age_days", "txn_count_24h",
            "new_receiver", "device_changed", "location_mismatch"],
    "card": ["amount", "amount_vs_avg", "txn_count_1h", "minutes_since_last",
             "distance_from_last_km", "travel_speed_kmh", "merchant_risk",
             "is_online", "is_foreign"],
}


def _find_model_file(kind):
    env_name = ENV_PATHS[kind]
    if os.environ.get(env_name):
        p = Path(os.environ[env_name])
        if not p.exists():
            raise FileNotFoundError(f"{env_name} points to a missing file: {p}")
        return p

    candidates = []
    for folder in MODEL_DIRS:
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            name = p.name.lower()
            if (p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS
                    and kind in name and not any(w in name for w in SKIP_WORDS)):
                candidates.append(p)

    if not candidates:
        searched = ", ".join(str(d) for d in MODEL_DIRS)
        raise FileNotFoundError(
            f"No {kind.upper()} model file found. Searched: {searched}. "
            f"Put the file there with '{kind}' in its name, or set {env_name}.")

    candidates.sort(key=lambda p: ("xgb" not in p.name.lower(), str(p)))
    return candidates[0]


def _unpack(obj):
    """Return (final_estimator, preprocessing_or_None, feature_names_or_None)."""
    features = None
    if isinstance(obj, dict):
        for key in ("features", "feature_names", "columns"):
            if obj.get(key) is not None:
                features = list(obj[key])
                break
        model = None
        for key in ("model", "clf", "classifier", "pipeline", "estimator"):
            if key in obj:
                model = obj[key]
                break
        if model is None:
            raise ValueError("Model file is a dict but has no 'model' key.")
        obj = model

    pre, final = None, obj
    if hasattr(obj, "steps"):  # sklearn Pipeline
        final = obj.steps[-1][1]
        pre = obj[:-1] if len(obj.steps) > 1 else None
    return final, pre, features


@lru_cache(maxsize=None)
def get_bundle(kind):
    path = _find_model_file(kind)
    print(f"[{kind}] loading model: {path}")

    if path.suffix.lower() in (".json", ".ubj"):
        from xgboost import XGBClassifier
        final = XGBClassifier()
        final.load_model(str(path))
        pre, features = None, None
    else:
        final, pre, features = _unpack(joblib.load(path))

    if features is None:
        features = getattr(final, "feature_names_in_", None)
        if features is not None:
            features = list(features)
    if features is None:
        try:
            features = list(final.get_booster().feature_names or [])
        except Exception:
            features = None
    if not features:
        features = DEFAULT_FEATURES[kind]

    explainer = shap.TreeExplainer(final)
    return {"path": path, "model": final, "pre": pre,
            "features": features, "explainer": explainer}


def score(kind, payload, threshold=0.5):
    """Score one transaction.

    kind:      "upi" or "card"
    payload:   dict of features (a pydantic model also works)
    threshold: probability at or above which the decision is FRAUD
    """
    if kind not in ENV_PATHS:
        raise ValueError(f"Unknown kind '{kind}'. Use 'upi' or 'card'.")

    b = get_bundle(kind)

    row = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
    missing = [f for f in b["features"] if f not in row]
    if missing:
        raise ValueError(f"Model expects features that were not provided: {missing}")

    X = pd.DataFrame([[row[f] for f in b["features"]]], columns=b["features"])
    X_in = b["pre"].transform(X) if b["pre"] is not None else X

    prob = float(b["model"].predict_proba(X_in)[0][1])

    sv = b["explainer"].shap_values(X_in)
    if isinstance(sv, list):          # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer shap: (n, features, classes)
        sv = sv[:, :, 1]
    impacts = sv[0]

    names = b["features"] if len(b["features"]) == len(impacts) else \
        [f"feature_{i}" for i in range(len(impacts))]

    reasons = [{"feature": n, "impact": float(v)} for n, v in zip(names, impacts)]
    reasons.sort(key=lambda x: x["impact"], reverse=True)  # most risk-increasing first

    return {
        "fraud_probability": prob,
        "decision": "FRAUD" if prob >= threshold else "NORMAL",
        "threshold": threshold,
        "top_reasons": reasons[:5],
    }