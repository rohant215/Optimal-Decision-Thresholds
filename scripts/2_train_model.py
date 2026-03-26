"""
FILE 2: train_model.py

Load cohort data, handle missing values, train logistic regression,
and save predicted probabilities.
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss

DATA_FILE   = "data/processed/cohort.csv"
OUTPUT_FILE = "data/processed/predictions.csv"
MODEL_FILE  = "data/model.pkl"


# Load data
print("Loading cohort data...")
df = pd.read_csv(DATA_FILE)

FEATURE_COLS = [
    "heart_rate", "sbp", "dbp", "resp_rate",
    "temp_c", "spo2", "glucose"
]
TARGET_COL = "died"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"  → {len(df):,} patients, {y.mean():.1%} mortality rate")


# Build pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf",     LogisticRegression(max_iter=1000, random_state=42))
])


# Cross-validated probabilities (out-of-fold)
print("Running 5-fold cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

probs = cross_val_predict(
    model, X, y,
    cv=cv,
    method="predict_proba"
)[:, 1]

print(f"  → Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
print(f"  → Mean predicted probability: {probs.mean():.3f} (true rate: {y.mean():.3f})")


# Evaluate
auroc = roc_auc_score(y, probs)
brier = brier_score_loss(y, probs)

print("\nModel performance (cross-validated):")
print(f"  AUROC:       {auroc:.4f}")
print(f"  Brier score: {brier:.4f}")


# Train final model
print("\nTraining final model...")
model.fit(X, y)

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"  → Model saved to {MODEL_FILE}")


# Save predictions
df["prob_cv"] = probs
df["true_label"] = y

df.to_csv(OUTPUT_FILE, index=False)
print(f"  → Predictions saved to {OUTPUT_FILE}")


# Probability distribution sanity check
print("\nProbability distribution:")
for threshold in [0.1, 0.2, 0.3, 0.5]:
    flagged = (probs >= threshold).mean()
    print(f"  Patients flagged at threshold {threshold}: {flagged:.1%}")