"""
Population Stability Index (PSI) drift monitoring.

Everything else in this project is a one-time backtest. A real deployment
needs ongoing monitoring: does the population you're scoring today still
look like the population the model was trained on? PSI is the standard
metric for this (see docs/PATH_TO_PRODUCTION.md, Section 7).

This project has no live production stream to monitor -- but it has a
genuine, real before/after split already: the time-based train/test split
(pre-2015 vs. 2015+, src/features.py). Comparing those two populations'
feature distributions is a real drift check, not a synthetic one -- if the
test population has drifted meaningfully from train, that's itself part of
the explanation for the model's modest performance (docs/MODEL_VALIDATION.md
Section 5).

Standard industry thresholds:
    PSI < 0.10           -- no significant population change
    0.10 <= PSI < 0.25    -- moderate shift, worth monitoring
    PSI >= 0.25           -- significant shift, worth investigating
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25


def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    PSI between two distributions of the same numeric feature. `expected`
    is the baseline (e.g. training population); `actual` is the population
    being checked for drift (e.g. a later time period). Bucket edges are
    derived from `expected`'s quantiles, so bucketing itself doesn't depend
    on `actual` (a real monitoring system wouldn't have `actual` in advance
    when setting up its bins).
    """
    expected = expected.dropna()
    actual = actual.dropna()

    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        # Degenerate case: expected has too little variation to bucket meaningfully.
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts = pd.cut(expected, bins=edges).value_counts(sort=False)
    actual_counts = pd.cut(actual, bins=edges).value_counts(sort=False)

    expected_pct = (expected_counts / len(expected)).clip(lower=1e-6)
    actual_pct = (actual_counts / len(actual)).clip(lower=1e-6)

    psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
    return float(psi)


def compute_psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for a categorical feature -- buckets are the categories themselves."""
    expected = expected.dropna()
    actual = actual.dropna()

    all_categories = set(expected.unique()) | set(actual.unique())
    expected_pct = expected.value_counts(normalize=True).reindex(all_categories, fill_value=0).clip(lower=1e-6)
    actual_pct = actual.value_counts(normalize=True).reindex(all_categories, fill_value=0).clip(lower=1e-6)

    psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
    return float(psi)


def classify_psi(psi: float) -> str:
    if psi < PSI_MODERATE_THRESHOLD:
        return "stable"
    elif psi < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate_shift"
    else:
        return "significant_shift"


def compute_psi_report(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        numeric_features: list[str], categorical_features: list[str],
                        buckets: int = 10) -> pd.DataFrame:
    """
    Runs PSI across every feature, numeric and categorical, comparing the
    training population (baseline) to the test population (the population
    being checked for drift). Returns one row per feature.
    """
    rows = []
    for col in numeric_features:
        psi = compute_psi(train_df[col], test_df[col], buckets=buckets)
        rows.append({"feature": col, "type": "numeric", "psi": psi, "status": classify_psi(psi)})

    for col in categorical_features:
        psi = compute_psi_categorical(train_df[col], test_df[col])
        rows.append({"feature": col, "type": "categorical", "psi": psi, "status": classify_psi(psi)})

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
