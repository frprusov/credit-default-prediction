"""
Feature engineering for the Home Credit dataset.

Each function takes a raw DataFrame and returns an aggregated DataFrame
with one row per SK_ID_CURR, ready to merge into application_train.
"""

import pandas as pd
import numpy as np


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Summarise the bureau table into one row per client."""
    agg = bureau.groupby('SK_ID_CURR').agg(
        bureau_count=('SK_ID_BUREAU', 'count'),
        bureau_active_count=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        bureau_closed_count=('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
        bureau_amt_credit_sum=('AMT_CREDIT_SUM', 'sum'),
        bureau_amt_credit_mean=('AMT_CREDIT_SUM', 'mean'),
        bureau_amt_credit_max=('AMT_CREDIT_SUM', 'max'),
        bureau_amt_debt_sum=('AMT_CREDIT_SUM_DEBT', 'sum'),
        bureau_overdue_max=('AMT_CREDIT_SUM_OVERDUE', 'max'),
        bureau_days_credit_max=('DAYS_CREDIT', 'max'),
        bureau_days_credit_min=('DAYS_CREDIT', 'min'),
    )
    agg['bureau_debt_credit_ratio'] = (
        agg['bureau_amt_debt_sum'] / agg['bureau_amt_credit_sum'].replace(0, np.nan)
    )
    return agg


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """Summarise the previous_application table into one row per client."""
    agg = prev.groupby('SK_ID_CURR').agg(
        prev_count=('SK_ID_PREV', 'count'),
        prev_approved_count=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
        prev_refused_count=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
        prev_amt_application_mean=('AMT_APPLICATION', 'mean'),
        prev_amt_application_max=('AMT_APPLICATION', 'max'),
        prev_amt_credit_mean=('AMT_CREDIT', 'mean'),
        prev_amt_credit_max=('AMT_CREDIT', 'max'),
        prev_days_decision_max=('DAYS_DECISION', 'max'),
        prev_days_decision_min=('DAYS_DECISION', 'min'),
    )
    agg['prev_refusal_rate'] = agg['prev_refused_count'] / agg['prev_count']
    agg['prev_credit_vs_application'] = (
        agg['prev_amt_credit_mean'] / agg['prev_amt_application_mean'].replace(0, np.nan)
    )
    return agg


def build_features(data_dir: str = "../data") -> tuple[pd.DataFrame, pd.Series]:
    """
    Load all tables, aggregate them, merge into the application table.
    Returns (X, y) ready for train/test split.
    """
    app = pd.read_csv(f"{data_dir}/application_train.csv")
    bureau = pd.read_csv(f"{data_dir}/bureau.csv")
    prev = pd.read_csv(f"{data_dir}/previous_application.csv")

    bureau_agg = aggregate_bureau(bureau)
    prev_agg = aggregate_previous_application(prev)

    df = app.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df = df.merge(prev_agg, on='SK_ID_CURR', how='left')

    feature_cols = [c for c in df.columns if c not in ['TARGET', 'SK_ID_CURR']]
    X = df[feature_cols]
    y = df['TARGET']

    return X, y