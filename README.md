# Credit Default Prediction

Predicting loan default risk on the Home Credit dataset, with a focus on
model interpretability, probability calibration, and fairness — not just
maximum AUC.

## Motivation

Banks need more than a ranking of "likely to default" — they need
calibrated probabilities to price loans, explanations to justify
decisions to customers and regulators, and evidence that the model
performs fairly across demographic groups. This project builds a
credit-default model end to end with those constraints in mind.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
— seven tables covering 307,511 loan applications, prior applications
with Home Credit, and previous credit history reported by other
institutions. Target: whether the client had payment difficulties.
Base default rate is 8.1%.

## Approach

Progressive improvement across notebooks, each measured against the
previous baseline:

| Step | Features                     | Model               | AUC-ROC |
|------|------------------------------|---------------------|---------|
| 1    | 10 numeric main-table only   | Logistic regression | 0.62    |
| 2    | + all categorical            | Logistic regression | 0.66    |
| 3    | + all main-table features    | Logistic regression | 0.74    |
| 4    | + bureau aggregations        | Logistic regression | 0.75    |
| 5    | + previous_application       | Logistic regression | 0.76    |
| 6    | Same features                | LightGBM            | in progress |

Deliberate choices worth noting:

- **A logistic-regression baseline before the boosted-tree model.** Establishes
  what the algorithm switch is worth. Without a baseline, an AUC of 0.77
  is a number without context.
- **Feature aggregation from multi-row tables to one row per client.** The
  bureau table has ~5 loans per client on average, summarised via
  count / sum / mean / max statistics plus derived ratios (debt-to-credit,
  refusal rate).
- **Preprocessing in a `ColumnTransformer` pipeline.** Ensures that
  imputation and scaling statistics are learned only on training data
  and applied consistently to test data.

## Planned work

- Aggregations for the remaining tables (`installments_payments`,
  `POS_CASH_balance`, `credit_card_balance`)
- Probability calibration with isotonic regression, evaluated with
  Brier score and reliability plots
- Feature-level and prediction-level explanations with SHAP
- Fairness analysis across age and gender with Fairlearn
- Cross-validation with honest error bars, not single-split point
  estimates
- Reproducible training script (`scripts/train.py`) and containerised
  serving

## Structure
