"""Shared fixtures for testing features.py and models.py against synthetic data."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_loans_df():
    """
    A synthetic dataset shaped like the OUTPUT of src.data_loader.clean_and_scope --
    i.e. already has fico_avg, term_months, int_rate_frac, emp_length_years,
    defaulted, etc. Used to test features.py and models.py without needing the
    real multi-GB Lending Club file.
    """
    rng = np.random.default_rng(42)
    n = 300

    fico_avg = rng.normal(670, 40, n).clip(580, 780)
    # Default probability decreases as fico_avg increases -- gives the model
    # something real to learn, so AUC/Gini/KS aren't degenerate on this fixture.
    pd_true = np.clip(1 / (1 + np.exp((fico_avg - 660) / 30)) * 0.5, 0.02, 0.6)
    defaulted = rng.random(n) < pd_true

    months = rng.integers(1, 145, n)  # 1-144 months after Jan 2007
    issue_dates = pd.Timestamp("2007-01-01") + pd.to_timedelta(months * 30, unit="D")

    df = pd.DataFrame({
        "fico_avg": fico_avg,
        "dti": rng.normal(20, 6, n).clip(1, 40),
        "annual_inc": rng.normal(50000, 15000, n).clip(15000, 150000),
        "emp_length_years": rng.choice([0, 1, 2, 3, 5, 10, np.nan], n, p=[0.1,0.1,0.1,0.1,0.2,0.3,0.1]),
        "term_months": rng.choice([36, 60], n),
        "loan_amnt": rng.normal(9000, 3000, n).clip(1000, 25000),
        "int_rate_frac": rng.normal(0.16, 0.05, n).clip(0.06, 0.30),
        "revol_util_frac": rng.normal(0.45, 0.2, n).clip(0.0, 1.0),
        "delinq_2yrs": rng.poisson(0.3, n).astype(float),
        "open_acc": rng.integers(2, 20, n).astype(float),
        "pub_rec": rng.poisson(0.1, n).astype(float),
        "inq_last_6mths": rng.poisson(0.8, n).astype(float),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other"], n),
        "defaulted": defaulted.astype(int),
        "issue_d": issue_dates.strftime("%b-%Y"),
    })
    return df
