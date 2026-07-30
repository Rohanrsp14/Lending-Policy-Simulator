import numpy as np
import pandas as pd
import pytest

from src.monitor import (
    PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
    classify_psi,
    compute_psi,
    compute_psi_categorical,
    compute_psi_report,
)


@pytest.fixture
def identical_distributions():
    rng = np.random.default_rng(42)
    data = rng.normal(650, 40, 5000)
    return pd.Series(data[:2500]), pd.Series(data[2500:])


@pytest.fixture
def shifted_distributions():
    rng = np.random.default_rng(42)
    expected = pd.Series(rng.normal(650, 40, 2500))
    actual = pd.Series(rng.normal(600, 40, 2500))  # meaningfully shifted mean
    return expected, actual


def test_compute_psi_near_zero_for_identical_distributions(identical_distributions):
    expected, actual = identical_distributions
    psi = compute_psi(expected, actual)
    assert psi < PSI_MODERATE_THRESHOLD


def test_compute_psi_detects_real_shift(shifted_distributions):
    expected, actual = shifted_distributions
    psi = compute_psi(expected, actual)
    assert psi >= PSI_MODERATE_THRESHOLD


def test_compute_psi_handles_nans():
    expected = pd.Series([1, 2, 3, 4, 5, np.nan] * 100)
    actual = pd.Series([1, 2, 3, 4, 5, np.nan] * 100)
    psi = compute_psi(expected, actual)
    assert psi < PSI_MODERATE_THRESHOLD


def test_compute_psi_degenerate_expected_returns_zero():
    # expected has almost no variation -- can't form meaningful quantile buckets
    expected = pd.Series([5] * 100)
    actual = pd.Series([5, 6, 7] * 30)
    psi = compute_psi(expected, actual)
    assert psi == 0.0


def test_compute_psi_categorical_identical():
    rng = np.random.default_rng(1)
    cats = rng.choice(["A", "B", "C"], 2000, p=[0.5, 0.3, 0.2])
    expected = pd.Series(cats[:1000])
    actual = pd.Series(cats[1000:])
    psi = compute_psi_categorical(expected, actual)
    assert psi < PSI_MODERATE_THRESHOLD


def test_compute_psi_categorical_detects_shift():
    expected = pd.Series(["A"] * 800 + ["B"] * 200)
    actual = pd.Series(["A"] * 200 + ["B"] * 800)  # flipped proportions
    psi = compute_psi_categorical(expected, actual)
    assert psi >= PSI_SIGNIFICANT_THRESHOLD


def test_compute_psi_categorical_handles_unseen_category():
    expected = pd.Series(["A"] * 500 + ["B"] * 500)
    actual = pd.Series(["A"] * 400 + ["B"] * 400 + ["C"] * 200)  # new category appears
    psi = compute_psi_categorical(expected, actual)
    assert psi > 0  # a brand-new category is itself a real shift


def test_classify_psi_thresholds():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate_shift"
    assert classify_psi(0.30) == "significant_shift"


def test_compute_psi_report_returns_one_row_per_feature():
    rng = np.random.default_rng(7)
    n = 1000
    train_df = pd.DataFrame({
        "fico_avg": rng.normal(670, 40, n),
        "dti": rng.normal(20, 6, n),
        "purpose": rng.choice(["debt_consolidation", "credit_card"], n),
    })
    test_df = pd.DataFrame({
        "fico_avg": rng.normal(670, 40, n),
        "dti": rng.normal(20, 6, n),
        "purpose": rng.choice(["debt_consolidation", "credit_card"], n),
    })
    report = compute_psi_report(train_df, test_df, ["fico_avg", "dti"], ["purpose"])
    assert len(report) == 3
    assert set(report["feature"]) == {"fico_avg", "dti", "purpose"}
    assert set(report["status"]).issubset({"stable", "moderate_shift", "significant_shift"})


def test_compute_psi_report_sorted_descending_by_psi():
    rng = np.random.default_rng(3)
    n = 1000
    train_df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    test_df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(5, 1, n)})  # b shifted hard
    report = compute_psi_report(train_df, test_df, ["a", "b"], [])
    assert report.iloc[0]["feature"] == "b"  # most-shifted feature first
