# Credit Default Prediction

Predicting loan default risk on the Home Credit dataset, with a focus on
probability calibration and model explainability — not just maximum AUC.

## Motivation

A credit risk model that only ranks applicants is not enough. A bank prices
loans using expected loss, which requires probabilities that mean what they
say. It has to justify rejections to customers and to regulators, which
requires per-decision explanations. And it has to demonstrate that the model
performs fairly across demographic groups.

This project builds a default prediction model end to end with those
constraints treated as first-class requirements rather than afterthoughts.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk):
307,511 loan applications across seven tables, covering the current
application, prior applications with Home Credit, and credit history reported
by other institutions. Target: whether the client had payment difficulties.
Base default rate is 8.1%.

## Results

Each step is measured against the previous one, so the contribution of every
addition is visible:

| Step | Features | Model | AUC-ROC |
|------|----------|-------|---------|
| 1 | 10 numeric main-table columns | Logistic regression | 0.6237 |
| 2 | + all categorical columns | Logistic regression | 0.6645 |
| 3 | + all main-table features | Logistic regression | 0.7488 |
| 4 | + bureau aggregations | Logistic regression | 0.7529 |
| 5 | + previous_application aggregations | Logistic regression | 0.7581 |
| 6 | same features | LightGBM | **0.7701** |

AUC-PR on the final model is 0.26, against a random baseline of 0.08.

The logistic regression baseline is not decoration. Without it, an AUC of 0.77
is a number with no reference point; with it, the 0.04 gain from switching to
gradient boosting is a measured result that justifies the added complexity.

## Calibration

![Reliability diagram](reports/calibration.png)

The raw LightGBM output is severely miscalibrated. At a predicted probability
of 0.75, the observed default rate is only about 0.24. The cause is
`class_weight="balanced"`, which up-weights the minority class during training
and inflates output probabilities far above the true 8% base rate. This helps
ranking but makes the scores unusable as probabilities.

The Brier score quantifies the problem. Predicting the base rate of 0.08 for
every client scores about 0.074, so the uncalibrated model at **0.16 performs
worse than a constant prediction** in probabilistic terms, despite its AUC of
0.77. After isotonic calibration on a held-out calibration set the Brier score
falls to **0.065**, below the constant baseline, while AUC is unchanged —
isotonic regression is monotonic, so it re-maps values without altering the
ranking.

## Explainability

![SHAP summary](reports/shap_summary.png)

The three `EXT_SOURCE` features — normalised scores from external credit
bureaus — dominate the model. Their direction is as expected: higher external
scores push predictions towards lower risk.

Three of the aggregated features built in `src/features.py` rank in the top
fifteen, indicating that the multi-table feature engineering added genuine
signal:

- **`prev_credit_vs_application`** (rank 4) — clients who previously received
  less credit than they applied for are predicted as higher risk.
- **`bureau_debt_credit_ratio`** (rank 8) — a higher share of outstanding debt
  relative to total credit increases predicted risk.
- **`prev_refusal_rate`** (rank 14) — a history of refused applications
  increases predicted risk.

`CODE_GENDER` also appears among the top features and contributes materially
to individual predictions. A model that uses gender as a predictor requires
explicit fairness assessment before deployment could be considered — group-wise
performance and error rates need to be measured, not assumed. This is the
motivation for the fairness analysis listed below.

## Method notes

- **Split before exploring.** The test set is held out before any EDA or
  feature selection, and touched only for final evaluation.
- **Preprocessing inside the pipeline.** Imputation and scaling statistics are
  learned on training data only and applied unchanged to the test set, making
  leakage through preprocessing structurally impossible.
- **Three-way split for calibration.** The calibration mapping is fitted on a
  separate held-out set the model never saw during training.
- **Aggregation before joining.** The auxiliary tables hold multiple rows per
  client, summarised to one row per client using counts, sums, means, maxima
  and derived ratios before merging.

## Structure

```
credit-default-prediction/
├── data/                         # not tracked — download from Kaggle
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_categorical.ipynb
│   ├── 03_all_main_features.ipynb
│   ├── 04_bureau_features.ipynb
│   ├── 05_previous_app_features.ipynb
│   ├── 06_lightgbm.ipynb
│   └── 07_calibration_shap.ipynb
├── src/
│   └── features.py               # multi-table aggregation and feature assembly
├── reports/                      # figures
└── README.md
```

## Reproducing

```bash
conda create -n credit python=3.11
conda activate credit
pip install pandas numpy scikit-learn matplotlib seaborn jupyter lightgbm shap
```

Download the competition data and unzip it into `data/`, then run the
notebooks in order.

## Planned work

- Fairness analysis with Fairlearn across age and gender, measuring group-wise
  AUC, false positive rates and selection rates
- Cross-validated error bars instead of single-split point estimates
- A reproducible training script (`train.py`) reproducing the final model from
  the command line

## Author

Francisak Prusov — MSc Electrical Engineering, Ghent University (2026).
Built as a self-directed project while moving into applied machine learning.