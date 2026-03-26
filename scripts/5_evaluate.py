"""
FILE 5: evaluate.py

Generate final metrics and publication-quality figures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

DATA_FILE    = "data/processed/calibrated.csv"
RESULTS_FILE = "results/threshold_results.csv"

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

df         = pd.read_csv(DATA_FILE)
results_df = pd.read_csv(RESULTS_FILE)
y          = df["true_label"].values

PROB_VERSIONS = {
    "Raw (uncalibrated)":  "prob_raw",
    "Platt scaling":       "prob_platt",
    "Isotonic regression": "prob_isotonic",
    "Overconfident":       "prob_overconf",
    "Underconfident":      "prob_underconf",
}

COLORS = {
    "Raw (uncalibrated)":  "#888780",
    "Platt scaling":       "#1D9E75",
    "Isotonic regression": "#7F77DD",
    "Overconfident":       "#D85A30",
    "Underconfident":      "#BA7517",
}


# Expected Calibration Error
def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    error = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        error += (mask.sum() / n) * abs(
            y_true[mask].mean() - y_prob[mask].mean()
        )
    return error


# Metrics table
print("Computing final metrics table...")

rows = []
for name, col in PROB_VERSIONS.items():
    p = df[col].values

    avg_bias = results_df[
        results_df["version"] == name
    ]["threshold_bias"].mean()

    avg_cost_increase = results_df[
        results_df["version"] == name
    ]["loss_increase_pct"].mean()

    rows.append({
        "Version":               name,
        "AUROC":                 round(roc_auc_score(y, p), 4),
        "Brier score":           round(brier_score_loss(y, p), 4),
        "ECE":                   round(ece(y, p), 4),
        "Avg threshold bias":    round(avg_bias, 4),
        "Avg cost increase (%)": round(avg_cost_increase, 2),
    })

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("results/final_metrics_table.csv", index=False)

print(metrics_df.to_string(index=False))


# Decision Curve Analysis
print("\nGenerating Decision Curve Analysis plot...")

def net_benefit(y_true, y_prob, thresholds):
    n = len(y_true)
    nb = []
    for t in thresholds:
        tp = ((y_true == 1) & (y_prob >= t)).sum()
        fp = ((y_true == 0) & (y_prob >= t)).sum()
        nb.append((tp / n) - (fp / n) * (t / (1 - t + 1e-9)))
    return np.array(nb)

thresholds = np.linspace(0.01, 0.50, 200)
prevalence = y.mean()

fig, ax = plt.subplots(figsize=(8, 5))

nb_treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds + 1e-9)
ax.plot(thresholds, nb_treat_all, color="black", linestyle="--",
        linewidth=1.5, label="Treat all")

ax.axhline(0, color="black", linestyle=":", linewidth=1.5, label="Treat none")

for name, col in PROB_VERSIONS.items():
    p = df[col].values
    nb = net_benefit(y, p, thresholds)
    ax.plot(thresholds, nb, color=COLORS[name], linewidth=2, label=name)

ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.set_title("Decision Curve Analysis")
ax.set_ylim(-0.02, prevalence * 1.2)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/decision_curves.png", dpi=150)
plt.close()

print("  → Saved figures/decision_curves.png")


# ECE vs threshold bias
print("Generating ECE vs threshold bias plot...")

fig, ax = plt.subplots(figsize=(7, 5))

for name, col in PROB_VERSIONS.items():
    p = df[col].values
    ece_val = ece(y, p)

    biases = results_df[
        results_df["version"] == name
    ]["threshold_bias"].values

    ax.scatter(
        [ece_val] * len(biases), biases,
        color=COLORS[name], s=60, label=name, zorder=3
    )

    ax.plot(
        [ece_val, ece_val],
        [biases.min(), biases.max()],
        color=COLORS[name], linewidth=1, alpha=0.5
    )

ax.set_xlabel("ECE")
ax.set_ylabel("Threshold bias")
ax.set_title("Calibration error vs threshold bias")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/calibration_error_vs_threshold_bias.png", dpi=150)
plt.close()

print("  → Saved figures/calibration_error_vs_threshold_bias.png")


print("\n✓ All evaluation complete.")