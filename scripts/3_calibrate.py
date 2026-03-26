"""
FILE 3: calibrate.py

Apply calibration methods, simulate miscalibration,
and save all probability versions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

DATA_FILE   = "data/processed/predictions.csv"
OUTPUT_FILE = "data/processed/calibrated.csv"
os.makedirs("plots", exist_ok=True)


# Load predictions
print("Loading predictions...")
df = pd.read_csv(DATA_FILE)
probs_raw = df["prob_cv"].values
y = df["true_label"].values


# Platt scaling (cross-validated)
print("Applying Platt scaling...")

probs_platt = np.zeros_like(probs_raw)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in cv.split(probs_raw, y):
    platt = LogisticRegression(max_iter=1000)
    platt.fit(probs_raw[train_idx].reshape(-1, 1), y[train_idx])
    probs_platt[val_idx] = platt.predict_proba(
        probs_raw[val_idx].reshape(-1, 1)
    )[:, 1]

print(f"  → Platt mean: {probs_platt.mean():.4f} (true rate: {y.mean():.4f})")


# Isotonic regression (cross-validated)
print("Applying isotonic regression...")

probs_isotonic = np.zeros_like(probs_raw)

for train_idx, val_idx in cv.split(probs_raw, y):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_raw[train_idx], y[train_idx])
    probs_isotonic[val_idx] = iso.predict(probs_raw[val_idx])

print(f"  → Isotonic mean: {probs_isotonic.mean():.4f} (true rate: {y.mean():.4f})")


# Controlled miscalibration
def miscalibrate_overconfident(p, strength=1.8):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    stretched = strength * logit_p
    return 1 / (1 + np.exp(-stretched))

def miscalibrate_underconfident(p, strength=0.5):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    shrunk = strength * logit_p
    return 1 / (1 + np.exp(-shrunk))

probs_overconf  = miscalibrate_overconfident(probs_raw,  strength=1.8)
probs_underconf = miscalibrate_underconfident(probs_raw, strength=0.5)

print(f"  → Overconfident mean:  {probs_overconf.mean():.4f}")
print(f"  → Underconfident mean: {probs_underconf.mean():.4f}")


# Reliability curves
print("Plotting reliability curves...")

fig, ax = plt.subplots(figsize=(7, 6))

versions = {
    "Raw":          (probs_raw,       "#888780", "-"),
    "Platt":        (probs_platt,     "#1D9E75", "-"),
    "Isotonic":     (probs_isotonic,  "#7F77DD", "-"),
    "Overconf":     (probs_overconf,  "#D85A30", "--"),
    "Underconf":    (probs_underconf, "#BA7517", "--"),
}

for label, (p, color, ls) in versions.items():
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, color=color, linestyle=ls,
            linewidth=2, label=label, marker="o", markersize=4)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Reliability curves")
ax.legend(loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/reliability_curves.png", dpi=150)
plt.close()

print("  → Saved plots/reliability_curves.png")


# Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_size = mask.sum()
        actual_rate = y_true[mask].mean()
        mean_pred   = y_prob[mask].mean()
        ece += (bin_size / n) * abs(actual_rate - mean_pred)

    return ece

print("\nECE (lower is better):")
for label, (p, _, _) in versions.items():
    ece = expected_calibration_error(y, p)
    print(f"  {label:12s}: {ece:.4f}")


# Save all probability versions
df["prob_raw"]       = probs_raw
df["prob_platt"]     = probs_platt
df["prob_isotonic"]  = probs_isotonic
df["prob_overconf"]  = probs_overconf
df["prob_underconf"] = probs_underconf

df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✓ Saved to {OUTPUT_FILE}")
print(f"  Columns: {[c for c in df.columns if c.startswith('prob_')]}")