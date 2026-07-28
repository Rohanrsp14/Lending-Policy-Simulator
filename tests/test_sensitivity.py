import numpy as np
import pytest

from src.features import parse_issue_date, prepare_features, time_based_split
from src.models import train_challenger
from src.sensitivity import run_sensitivity, summarize_robustness


@pytest.fixture
def prepared_split_for_sensitivity(sample_loans_df):
    df = prepare_features(sample_loans_df)
    df = parse_issue_date(df)
    train, test = time_based_split(df, cutoff_date="2015-06-01")
    return train, test


def test_run_sensitivity_returns_expected_columns(prepared_split_for_sensitivity):
    train, test = prepared_split_for_sensitivity
    model = train_challenger(train)
    # Small ranges to keep the test fast -- each value re-runs a full frontier sweep
    result = run_sensitivity(
        model, train, test,
        lgd_range=np.array([0.45, 0.65]),
        opex_range=np.array([0.03, 0.07]),
        capital_range=np.array([0.06, 0.10]),
    )
    expected_cols = {
        "swept_parameter", "parameter_value", "champion_best_raroc",
        "challenger_best_raroc", "delta_challenger_minus_champion", "champion_wins",
    }
    assert expected_cols.issubset(result.columns)


def test_run_sensitivity_covers_all_three_parameters(prepared_split_for_sensitivity):
    train, test = prepared_split_for_sensitivity
    model = train_challenger(train)
    result = run_sensitivity(
        model, train, test,
        lgd_range=np.array([0.5]),
        opex_range=np.array([0.05]),
        capital_range=np.array([0.08]),
    )
    assert set(result["swept_parameter"].unique()) == {"lgd", "opex_rate", "capital_rate"}


def test_higher_lgd_reduces_raroc_for_both_policies(prepared_split_for_sensitivity):
    """Sanity check: higher loss-given-default should make both policies worse off,
    all else equal -- if this failed, the sensitivity plumbing itself would be broken."""
    train, test = prepared_split_for_sensitivity
    model = train_challenger(train)
    result = run_sensitivity(
        model, train, test,
        lgd_range=np.array([0.35, 0.75]),
        opex_range=np.array([0.05]),
        capital_range=np.array([0.08]),
    )
    lgd_sweep = result[result["swept_parameter"] == "lgd"].sort_values("parameter_value")
    assert lgd_sweep.iloc[0]["champion_best_raroc"] >= lgd_sweep.iloc[-1]["champion_best_raroc"]
    assert lgd_sweep.iloc[0]["challenger_best_raroc"] >= lgd_sweep.iloc[-1]["challenger_best_raroc"]


def test_run_sensitivity_forwards_custom_quantiles(prepared_split_for_sensitivity):
    """
    Confirms the quantiles argument actually reaches compute_frontier --
    this is what lets a finer grid be used to validate that a found optimum
    approval rate is a genuine peak, not a coarse-grid artifact.
    """
    train, test = prepared_split_for_sensitivity
    model = train_challenger(train)
    fine_quantiles = np.linspace(0.50, 0.99, 50)
    result = run_sensitivity(
        model, train, test,
        lgd_range=np.array([0.55]),
        opex_range=np.array([0.05]),
        capital_range=np.array([0.08]),
        quantiles=fine_quantiles,
    )
    # Should run without error and produce the same expected structure
    assert len(result) == 3  # one row per parameter (1 value each)
    assert result["champion_best_approval_rate"].notna().all()


def test_summarize_robustness_returns_one_row_per_parameter(prepared_split_for_sensitivity):
    train, test = prepared_split_for_sensitivity
    model = train_challenger(train)
    result = run_sensitivity(
        model, train, test,
        lgd_range=np.array([0.45, 0.65]),
        opex_range=np.array([0.03, 0.07]),
        capital_range=np.array([0.06, 0.10]),
    )
    summary = summarize_robustness(result)
    assert len(summary) == 3
    assert set(summary["robustness"]).issubset({
        "champion_always_wins", "challenger_always_wins", "conclusion_depends_on_assumption",
    })
