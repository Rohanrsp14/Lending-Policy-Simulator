"""
Feature engineering for the Lending Policy Simulator.

Takes the cleaned output of src/data_loader.py and prepares it for both the
champion (rule-based cutoff) and challenger (trained PD model) policies:
parses issue_d into a real date for time-based splitting, and defines the
feature set the challenger model trains on.

See CLAUDE.md for why a time-based split is mandatory here (loan performance
and applicant mix both drift over 2007-2018; a random split would leak
future information into the training set).
"""
from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "fico_avg",
    "dti",
    "annual_inc",
    "emp_length_years",
    "term_months",
    "loan_amnt",
    "int_rate_frac",
    "revol_util_frac",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "inq_last_6mths",
    "mort_acc",
    "total_acc",
]
CATEGORICAL_FEATURES = ["purpose", "home_ownership", "verification_status"]
TARGET = "defaulted"


def parse_issue_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parses issue_d (e.g. 'Jan-2018') into a real datetime column, issue_date."""
    df = df.copy()
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    return df


def time_based_split(df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits into train (issue_date < cutoff_date) and test (issue_date >= cutoff_date).

    Locked as a hard requirement (not optional) -- a random split would mix future
    loan vintages into training and overstate how well the model would have performed
    at decision time. See CLAUDE.md.
    """
    if "issue_date" not in df.columns:
        df = parse_issue_date(df)

    cutoff = pd.Timestamp(cutoff_date)
    train = df[df["issue_date"] < cutoff].copy()
    test = df[df["issue_date"] >= cutoff].copy()
    return train, test


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleanup pass before modeling: fills missing emp_length_years with the
    population median (a stated, simple imputation choice -- not a silent one)
    and ensures the feature columns used downstream actually exist.
    """
    df = df.copy()
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"prepare_features: missing expected columns: {missing}")

    if df["emp_length_years"].isna().any():
        median_emp_length = df["emp_length_years"].median()
        df["emp_length_years"] = df["emp_length_years"].fillna(median_emp_length)

    if df["revol_util_frac"].isna().any():
        median_revol_util = df["revol_util_frac"].median()
        df["revol_util_frac"] = df["revol_util_frac"].fillna(median_revol_util)

    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna("UNKNOWN")

    return df
