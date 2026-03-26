"""
FILE 4: threshold_analysis.py

Compute optimal thresholds, measure bias, and evaluate clinical loss
across different calibration versions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "data/processed/calibrated.csv"
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# Load data
df = pd.read_csv(DATA_FILE)
y  = df["true_label"].values

PROB_VERSIONS = {
    "Raw (uncalibrated)":  "prob_raw",
    "Platt scaling":       "prob_platt",
    "Isotonic regression": "prob_isotonic",
    "Overconfident":       "prob_overconf",
    "Underconfident":      "prob_underconf",
}


# Optimal threshold formula
def compute_optimal_threshold(cost_fn, cost_fp):
    return cost_fp / (cost_fp + cost_fn)


# Expected clinical loss
def expected_loss(y_true, y_prob, threshold, cost_fn, cost_fp):
    predicted_pos = y_prob >= threshold
    predicted_neg = y_prob <  threshold

    fn = ((y_true == 1) & predicted_neg).sum()
    fp = ((y_true == 0) & predicted_pos).sum()

    n = len(y_true)
    return (cost_fn * fn + cost_fp * fp) / n


def find_empirical_optimal_threshold(y_true, y_prob, cost_fn, cost_fp):
    candidate_thresholds = np.unique(y_prob)
    best_t, best_loss = 0, np.inf

    for t in candidate_thresholds:
        loss = expected_loss(y_true, y_prob, t, cost_fn, cost_fp)
        if loss < best_loss:
            best_loss = loss
            best_t = t

    return best_t, best_loss


# Run analysis
print("Running threshold analysis...\n")

COST_RATIOS = [3, 5, 9, 14, 19]
results = []

for ratio in COST_RATIOS:
    cost_fn = ratio
    cost_fp = 1
    t_star = compute_optimal_threshold(cost_fn, cost_fp)

    for version_name, prob_col in PROB_VERSIONS.items():
        p = df[prob_col].values

        loss_at_tstar = expected_loss(y, p, t_star, cost_fn, cost_fp)

        t_empirical, loss_at_empirical = find_empirical_optimal_threshold(
            y, p, cost_fn, cost_fp
        )

        threshold_bias = abs(t_empirical - t_star)

        loss_increase_pct = 100 * (
            loss_at_tstar - loss_at_empirical
        ) / (loss_at_empirical + 1e-9)

        results.append({
            "cost_ratio":         ratio,
            "t_star":             round(t_star, 4),
            "version":            version_name,
            "t_empirical":        round(t_empirical, 4),
            "threshold_bias":     round(threshold_bias, 4),
            "loss_at_tstar":      round(loss_at_tstar, 4),
            "loss_at_empirical":  round(loss_at_empirical, 4),
            "loss_increase_pct":  round(loss_increase_pct, 2),
        })

results_df = pd.DataFrame(results)
results_df.to_csv("results/threshold_results.csv", index=False)

print(results_df.to_string(index=False))


# Cost vs threshold plot
print("\nGenerating cost vs threshold plot...")

ratio = 9
cost_fn, cost_fp = ratio, 1
t_star = compute_optimal_threshold(cost_fn, cost_fp)

thresholds = np.linspace(0.01, 0.99, 200)

fig, ax = plt.subplots(figsize=(8, 5))

colors = {
    "Raw (uncalibrated)":  "#888780",
    "Platt scaling":       "#1D9E75",
    "Isotonic regression": "#7F77DD",
    "Overconfident":       "#D85A30",
    "Underconfident":      "#BA7517",
}
linestyles = {
    "Raw (uncalibrated)":  "-",
    "Platt scaling":       "-",
    "Isotonic regression": "-",
    "Overconfident":       "--",
    "Underconfident":      "--",
}

for version_name, prob_col in PROB_VERSIONS.items():
    p = df[prob_col].values
    losses = [expected_loss(y, p, t, cost_fn, cost_fp) for t in thresholds]
    ax.plot(thresholds, losses,
            color=colors[version_name],
            linestyle=linestyles[version_name],
            linewidth=2, label=version_name)

ax.axvline(t_star, color="black", linestyle=":", linewidth=1.5)

ax.set_xlabel("Decision threshold")
ax.set_ylabel("Expected clinical loss")
ax.set_title(f"Expected loss vs threshold (ratio = {ratio})")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/cost_vs_threshold.png", dpi=150)
plt.close()

print("  → Saved figures/cost_vs_threshold.png")


# Threshold bias summary plot
print("Generating threshold bias summary plot...")

fig, axes = plt.subplots(1, len(COST_RATIOS), figsize=(14, 5), sharey=True)

for ax, ratio in zip(axes, COST_RATIOS):
    subset = results_df[results_df["cost_ratio"] == ratio]
    biases = [
        subset[subset["version"] == v]["threshold_bias"].values[0]
        for v in PROB_VERSIONS.keys()
    ]
    short_names = ["Raw", "Platt", "Isotonic", "Overconf.", "Underconf."]
    bar_colors  = ["#888780", "#1D9E75", "#7F77DD", "#D85A30", "#BA7517"]

    ax.bar(short_names, biases, color=bar_colors, width=0.6)
    ax.set_title(f"Ratio {ratio}:1\n(t*={1/(1+ratio):.2f})")
    ax.set_ylim(0, 0.15)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)

axes[0].set_ylabel("Threshold bias")

plt.tight_layout()
plt.savefig("figures/threshold_bias_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("  → Saved figures/threshold_bias_summary.png")


print("\n✓ Analysis complete")