# Optimal Decision Thresholds for Clinical Risk Prediction Models Under Calibration Uncertainty

A decision-theoretic analysis using MIMIC-III ICU data

---

## Overview

Clinical prediction models produce probabilities, but clinical practice requires binary decisions such as whether to treat or not. This conversion is governed by a decision threshold. In most applications, this threshold is either set arbitrarily or chosen using classification metrics such as the F1 score or Youden’s index.

This repository investigates a central issue that is often overlooked in applied machine learning:

**When predicted probabilities are miscalibrated, the decision threshold derived from them becomes incorrect, leading to suboptimal and potentially harmful decisions.**

Using a cohort of 55,653 ICU patients from the MIMIC-III database, this project provides both a theoretical and empirical analysis of how calibration error affects threshold selection and downstream clinical outcomes.

---

## Key Idea

Under asymmetric misclassification costs, the Bayes-optimal decision threshold is:

```
t* = C_FP / (C_FP + C_FN)
```

This result assumes that predicted probabilities are well calibrated.

When this assumption fails:

- The empirically optimal threshold shifts away from t*
- The magnitude of this shift can be substantial
- Expected clinical loss increases as a direct consequence

---

## Contributions

- Formal link between calibration error and threshold bias  
- Empirical evaluation across five probability variants and five cost ratios  
- Quantification of expected clinical loss increase  
- Evidence that underconfident models can be more harmful than overconfident ones  
- Comparison of Platt scaling and isotonic regression  

---

## Dataset

- Source: MIMIC-III ICU database  
- Cohort size: 55,653 patients  
- Outcome: in-hospital mortality  
- Features: seven vital signs from the first 24 hours  

---

## Repository Structure

```

|- data/
    |- raw/
        |- ADMISSIONS.csv
        |- CHARTEVENTS.csv
        |- ICUSTAYS.csv
    |- processed/
        |- ADMISSIONS.csv
        |- CHARTEVENTS.csv
        |- ICUSTAYS.csv
    |- model.pkl
|- figures/
    |- calibration_error_vs_threshold_bias.png
    |- cost_vs_threshold.png
    |- decision_curves.png
    |- reliability_curves.png
    |- threshold_bias_summary.png
|- results/
    |- final_metrics_table.csv
    |- threshold_results.csv
|- scripts/
    |- 01_load_data.py
    |- 2_train_model.py
    |- 3_calibrate.py
    |- 4_threshold_analysis.py
    |- 5_evaluate.py

```

---

## Pipeline

Run the scripts in order:

```bash
python scripts/01_load_data.py
python scripts/2_train_model.py
python scripts/3_calibrate.py
python scripts/4_threshold_analysis.py
python scripts/5_evaluate.py
```

---

## Results

### Reliability (Calibration)

![Reliability Curves](figures/reliability_curves.png)

Calibrated models closely follow the ideal diagonal, while overconfident and underconfident models show systematic deviation. Isotonic regression achieves the lowest calibration error.

---

### Expected Loss vs Threshold (Cost Ratio 9:1)

![Cost vs Threshold](figures/cost_vs_threshold.png)

Calibrated models achieve minimum loss near the theoretical threshold (0.10). Miscalibrated models are clearly shifted:

- Overconfident optimum near ~0.02  
- Underconfident optimum near ~0.26  

---

### Threshold Bias Across Cost Ratios

![Threshold Bias Summary](figures/threshold_bias_summary.png)

Miscalibrated models show consistently larger threshold bias across all cost ratios.

At cost ratio 9:1:

- Platt scaling: 0.0022  
- Overconfident: 0.0756  

---

### Calibration Error vs Threshold Bias

![ECE vs Bias](figures/calibration_error_vs_threshold_bias.png)

A clear positive relationship emerges:

**Higher calibration error leads directly to larger threshold bias**

This is the central empirical result of the project.

---

### Decision Curve Analysis

![Decision Curves](figures/decision_curves.png)

- Calibrated models maintain positive net benefit  
- Miscalibrated models can produce negative net benefit  
- This implies potential clinical harm if deployed  

---

### Quantitative Results

At cost ratio 9:1:

- Overconfident model: +20.2% increase in expected clinical loss  
- Underconfident model: +33.0% increase  
- Platt scaling: <1% increase  
- Isotonic regression: <1% increase  

At cost ratio 5:1:

- Underconfident model increases loss by over 100%  

---

## Key Takeaways

- Calibration directly determines decision quality  
- Threshold selection must follow calibration  
- Miscalibration leads to systematic decision errors  
- Underconfidence can be particularly dangerous in clinical settings  

---
