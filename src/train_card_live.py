import joblib
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("data/card_live_synthetic.csv")
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                      scale_pos_weight=ratio, eval_metric="aucpr", random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print("Precision:", round(precision_score(y_test, pred), 3))
print("Recall   :", round(recall_score(y_test, pred), 3))
print("F1       :", round(f1_score(y_test, pred), 3))
print("PR-AUC   :", round(average_precision_score(y_test, proba), 3))

joblib.dump(model, "models/xgb_card_live.pkl")
print("Saved models/xgb_card_live.pkl")