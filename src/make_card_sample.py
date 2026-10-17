import os

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# notebook lo vadina same split (random_state=42), so ivi model chudani rows
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

test = X_test.copy()
test["Class"] = y_test
fraud = test[test["Class"] == 1]
normal = test[test["Class"] == 0].sample(400, random_state=42)
sample = pd.concat([fraud, normal]).sample(frac=1, random_state=42)

os.makedirs("sample", exist_ok=True)
sample.to_csv("sample/card_sample.csv", index=False)
print(sample.shape, "fraud rows:", int(sample["Class"].sum()))