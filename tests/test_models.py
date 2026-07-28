import pandas as pd
import pytest

from src.features import prepare_features, time_based_split
from src.models import (
    CHAMPION_FICO_CUTOFF,
    calibrate_fico_cutoff,
    calibrate_pd_threshold,
    champion_decision,
    champion_decision_volume_matched,
    challenger_decision,
    challenger_decision_volume_matched,
    compute_raroc,
    evaluate_challenger,
    train_challenger,
)


@pytest.fixture
def prepared_split(sample_loans_df):
    df = prepare_features(sample_loans_df)
    from src.features import parse_issue_date
    df = parse_issue_date(df)
    train, test = time_based_split(df, cutoff_date="2013-01-01")
    return train, test


def test_champion_decision_respects_cutoff(sample_loans_df):
    approved = champion_decision(sample_loans_df, cutoff=700)
    assert (sample_loans_df.loc[approved, "fico_avg"] >= 700).all()
    assert (sample_loans_df.loc[~approved, "fico_avg"] < 700).all()


def test_champion_decision_default_cutoff_matches_constant(sample_loans_df):
    approved = champion_decision(sample_loans_df)
    assert (sample_loans_df.loc[approved, "fico_avg"] >= CHAMPION_FICO_CUTOFF).all()


def test_train_challenger_fits_without_error(prepared_split):
    train, _ = prepared_split
    model = train_challenger(train)
    assert model is not None


def test_challenger_decision_returns_boolean_series(prepared_split):
    train, test = prepared_split
    model = train_challenger(train)
    decision = challenger_decision(model, test, pd_threshold=0.3)
    assert decision.dtype == bool
    assert len(decision) == len(test)


def test_compute_raroc_empty_approval_returns_zeros(sample_loans_df):
    no_one_approved = pd.Series([False] * len(sample_loans_df), index=sample_loans_df.index)
    result = compute_raroc(sample_loans_df, no_one_approved)
    assert result["n"] == 0
    assert result["raroc"] == 0.0


def test_compute_raroc_uses_real_outcomes_not_model(sample_loans_df):
    all_approved = pd.Series([True] * len(sample_loans_df), index=sample_loans_df.index)
    result = compute_raroc(sample_loans_df, all_approved)
    # loss should be computable directly from realized 'defaulted' column
    expected_loss = (sample_loans_df["loan_amnt"] * sample_loans_df["defaulted"] * 0.55).sum()
    assert result["actual_loss"] == pytest.approx(expected_loss)


def test_compute_raroc_tighter_cutoff_changes_approval_rate(sample_loans_df):
    loose = champion_decision(sample_loans_df, cutoff=600)
    tight = champion_decision(sample_loans_df, cutoff=720)
    loose_metrics = compute_raroc(sample_loans_df, loose)
    tight_metrics = compute_raroc(sample_loans_df, tight)
    assert tight_metrics["approval_rate"] <= loose_metrics["approval_rate"]


def test_evaluate_challenger_returns_expected_keys(prepared_split):
    train, test = prepared_split
    model = train_challenger(train)
    metrics = evaluate_challenger(model, test)
    assert set(metrics.keys()) == {"auc", "gini", "ks"}
    assert 0.0 <= metrics["auc"] <= 1.0
    assert -1.0 <= metrics["gini"] <= 1.0
    assert 0.0 <= metrics["ks"] <= 1.0


def test_calibrate_fico_cutoff_produces_target_approval_rate(prepared_split):
    train, _ = prepared_split
    cutoff = calibrate_fico_cutoff(train, target_approval_rate=0.85)
    approved = champion_decision(train, cutoff=cutoff)
    # Should be close to 85% approval on the population it was calibrated on
    assert approved.mean() == pytest.approx(0.85, abs=0.03)


def test_champion_volume_matched_is_binding_regardless_of_scale(prepared_split):
    train, test = prepared_split
    # Even if every loan in `test` already clears a low absolute cutoff (the
    # reject-inference scenario seen on the real dataset), volume-matched
    # champion should still decline roughly 15% of the calibration population.
    approved_on_train = champion_decision_volume_matched(train, train, target_approval_rate=0.85)
    assert approved_on_train.mean() == pytest.approx(0.85, abs=0.03)


def test_calibrate_pd_threshold_produces_target_approval_rate(prepared_split):
    train, _ = prepared_split
    model = train_challenger(train)
    threshold = calibrate_pd_threshold(model, train, target_approval_rate=0.85)
    approved = challenger_decision(model, train, pd_threshold=threshold)
    assert approved.mean() == pytest.approx(0.85, abs=0.03)


def test_champion_and_challenger_volume_matched_are_comparable(prepared_split):
    train, test = prepared_split
    model = train_challenger(train)
    champ = champion_decision_volume_matched(test, train, target_approval_rate=0.85)
    chall = challenger_decision_volume_matched(model, test, train, target_approval_rate=0.85)
    # Both should approve roughly the same volume -- the whole point of a swap-set analysis
    assert abs(champ.mean() - chall.mean()) < 0.06
