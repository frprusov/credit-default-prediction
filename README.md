# Credit Default Prediction

Predicting loan default risk on the Home Credit dataset, with a focus on
probability calibration and model explainability — not just maximum AUC.

## Motivation

Machine learning is no longer something only software engineers need to
understand. As an electrical engineer it has become too central to the systems I
work on, and to daily life, to treat as someone else's specialism. I took a
course on Artificial Intelligence during my bachelor's, covering neural networks
and reinforcement learning, and this project is where I refresh that foundation
and extend it with the parts a course does not cover: real data, real class
imbalance, and the decisions that come after a model produces a number.

I deliberately chose a domain my engineering programme never touched. Credit
risk sits in the financial sector, it has consequences for the people it scores,
and it is regulated — which makes it a better teacher than another signal
processing problem would have been.

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

Each step is measured against the previous one on a single 80/20 split, so the
contribution of every addition is visible:

| Step | Features | Model | AUC-ROC |
|------|----------|-------|---------|
| 1 | 10 numeric main-table columns | Logistic regression | 0.6237 |
| 2 | + all categorical columns | Logistic regression | 0.6645 |
| 3 | + all main-table features | Logistic regression | 0.7488 |
| 4 | + bureau aggregations | Logistic regression | 0.7529 |
| 5 | + previous_application aggregations | Logistic regression | 0.7581 |
| 6 | same features | LightGBM | **0.7701** |

AUC-PR on the final model is 0.25, against a random baseline of 0.08.

Most of the gain comes from step 3: using every column of the main table rather
than a hand-picked ten is worth 0.084, far more than either set of multi-table
aggregations. Those add 0.009 between them — real, and visible again in the SHAP
rankings, but a reminder that the cheapest win was simply not discarding data.

### Cross-validated comparison

![Cross-validated AUC](reports/cv_comparison.png)

Single-split numbers carry no error bars, so the final two rows were re-measured
with 5-fold stratified cross-validation over the full dataset, both models sharing
the same folds:

| Model | AUC-ROC |
|-------|---------|
| Logistic regression | 0.7545 ± 0.0042 |
| LightGBM | 0.7668 ± 0.0036 |
| **Paired gain** | **0.0122 ± 0.0013**, LightGBM ahead in 5/5 folds |

The cross-validated gain of 0.0122 matches the single-split gap of 0.0120 almost
exactly, so the headline comparison holds up — but that agreement was not knowable
beforehand. The fold-to-fold spread of each model is 0.004, which is a third of
the effect being claimed: a single split could plausibly have landed anywhere
between 0.008 and 0.016, and there would have been no way to tell from one number.

The paired difference is the sharper instrument. Because both models saw identical
folds, subtracting within a fold cancels the variation caused by the split itself,
and the standard deviation of the difference (0.0013) is three times smaller than
that of either model's own score. Fold 1 is hard for both models and fold 2 easy
for both; only the model effect survives the subtraction. The result — a gain nine
times its own spread, consistent in every fold — is a stronger claim than either
mean could support on its own.

## Calibration

![Reliability diagram](reports/calibration.png)

The raw LightGBM output is severely miscalibrated. At a predicted probability
of 0.75, the observed default rate is only about 0.24. The cause is
`class_weight="balanced"`, which up-weights the minority class during training
and inflates output probabilities far above the true 8% base rate. This helps
ranking but makes the scores unusable as probabilities.

The Brier score quantifies the problem. Predicting the base rate of 0.081 for
every client scores 0.0742, so the uncalibrated model at **0.1651 performs worse
than a constant prediction** in probabilistic terms, despite its AUC of 0.77.
After isotonic calibration on a held-out calibration set the Brier score falls to
**0.0671**, below the constant baseline.

AUC moves from 0.7701 to 0.7672 in the process. Isotonic regression is monotonic
*non-decreasing*, not strictly increasing: it is a step function, so clients whose
raw scores differ slightly are mapped to an identical calibrated probability. Those
ties count as half a point each in the AUC, which costs 0.003. That is the price of
calibration, and at this magnitude it is worth paying — the ranking is essentially
preserved while the probabilities become usable for pricing.

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
motivation for the fairness analysis below.

## Fairness

![Group metrics](reports/fairness.png)

Every number here follows from one choice: a decision threshold of 0.15 on the
calibrated probability, giving an overall rejection rate of 16.3% against a true
default rate of 8.1%. That threshold is a business decision, not a property of the
model, and every disparity below scales with it.

| Group | n | True default rate | Rejection rate | FPR | FNR | AUC |
|-------|---|-------------------|----------------|-----|-----|-----|
| Female | 40,561 | 0.070 | 0.129 | 0.107 | 0.579 | 0.758 |
| Male | 20,940 | 0.102 | 0.228 | 0.191 | 0.443 | 0.767 |
| Age <30 | 8,905 | 0.113 | 0.290 | 0.250 | 0.401 | 0.746 |
| Age 30–40 | 16,491 | 0.096 | 0.205 | 0.169 | 0.456 | 0.771 |
| Age 40–50 | 15,299 | 0.076 | 0.150 | 0.124 | 0.539 | 0.764 |
| Age 50–60 | 13,620 | 0.063 | 0.100 | 0.083 | 0.652 | 0.752 |
| Age 60+ | 7,188 | 0.050 | 0.054 | 0.045 | 0.773 | 0.727 |

A third gender category holds two clients and supports no conclusion; because
Fairlearn computes `difference()` against the smallest group, the reported
aggregate gender differences are not the male–female gaps discussed here.

**The rejection gap is larger than the risk gap.** True default rates do differ
between groups, but the model amplifies those differences rather than tracking
them:

| Comparison | Ratio of true default rates | Ratio of rejection rates |
|------------|-----------------------------|--------------------------|
| Male vs female | 10.2% / 7.0% = 1.45 | 22.8% / 12.9% = 1.77 |
| Age <30 vs 60+ | 11.3% / 5.0% = 2.27 | 29.0% / 5.4% = 5.32 |

The same effect read down the age column: every group is rejected more often than
its own default rate, but the multiple falls monotonically from 2.6× for the
under-30s to 1.1× for the over-60s. Young applicants absorb the threshold, older
ones barely feel it — which is why the FNR at 60+ reaches 0.773 and the model
waves through more than three in four defaulters in that group.

This is not a defect to be patched. A fixed threshold on a monotone score
necessarily amplifies base-rate differences, because it cuts into the tail of the
score distribution, where a small shift in the group mean moves a large share of
the mass across the cut.

**The disparity is in outcomes, not in model quality.** AUC stays within
0.727–0.771 across every group, so the model separates defaulters from
non-defaulters about equally well everywhere. The unequal rejection rates come
from groups sitting at different points of the same score distribution under one
global threshold.

**Equal selection rates and equal error rates cannot both hold.** Across age
groups FPR falls from 0.250 to 0.045 while FNR rises from 0.401 to 0.773: groups
rejected more often absorb more false rejections and fewer missed defaults. When
true default rates differ between groups, no single threshold equalises selection
rate, false positive rate and false negative rate simultaneously. Which one to
equalise is a policy decision that determines who bears the cost of the model's
errors.

Before deployment this model would need `CODE_GENDER` removed and the cost in AUC
measured, group metrics reported across a range of thresholds rather than one,
and any fairness constraint applied explicitly — with a stated definition —
rather than left implicit.

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
- **Shared folds for model comparison.** Both models are cross-validated against
  one `StratifiedKFold` object, so the per-fold difference cancels the variation
  caused by the split and isolates the model effect.
- **One entry point.** `train.py` rebuilds the final model from the raw CSVs in a
  single command, so the result does not depend on the execution order of any
  notebook.

## Structure

```
credit-default-prediction/
├── train.py                      # reproduces the final model end to end
├── data/                         # not tracked — download from Kaggle
├── artifacts/                    # not tracked — written by train.py
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_categorical.ipynb
│   ├── 03_all_main_features.ipynb
│   ├── 04_bureau_features.ipynb
│   ├── 05_previous_app_features.ipynb
│   ├── 06_lightgbm.ipynb
│   ├── 07_calibration_shap.ipynb
│   ├── 08_fairness.ipynb
│   └── 09_cross_validation.ipynb
├── src/
│   └── features.py               # multi-table aggregation and feature assembly
├── reports/                      # figures
└── README.md
```

## Reproducing

```bash
conda create -n credit python=3.11
conda activate credit
pip install pandas numpy scikit-learn matplotlib seaborn jupyter \
            lightgbm shap fairlearn joblib
```

Download the competition data and unzip it into `data/`. Then either run the
notebooks in order to follow the progression, or reproduce the final model
directly:

```bash
python train.py --data-dir data --out-dir artifacts
```

This builds the features, fits and calibrates the model, and writes
`model.joblib`, `metrics.json` and a reliability diagram to `artifacts/`.

## Limitations

What this model does not do, stated plainly rather than left for a reader to
discover:

- **It uses `CODE_GENDER` as a predictor.** That is defensible in a study and not
  in production. Removing it and measuring the cost in AUC is the first change any
  deployment would require.
- **The decision threshold was chosen, not derived.** A threshold of 0.15 implies
  that a default costs about 5.7 times what a performing loan earns. In a bank that
  ratio comes from loss given default and margin; here it is an assumption, and
  every disparity reported above depends on it.
- **The training data only contains approved applicants.** Anyone refused credit in
  the past has no outcome recorded, so the model learns from a population already
  filtered by earlier decisions — including whatever bias those decisions carried.
- **Fairness cannot be checked on race or ethnicity**, because those attributes are
  not in the dataset. The analysis covers gender and age; on other protected
  attributes the model is simply unmeasured.
- **Hyperparameters were not tuned.** The LightGBM settings are reasonable defaults.
  A search would likely add a little AUC and would not change any conclusion here.
- **Calibration is measured on one split.** Unlike the AUC comparison, the Brier
  improvement has no error bars; cross-validating it needs a nested scheme.

## Author

Francisak Prusov — MSc Electrical Engineering, Ghent University (2026).
Built as a self-directed project while moving into applied machine learning.