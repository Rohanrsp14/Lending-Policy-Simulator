import pandas as pd
import pytest

from src.features import parse_issue_date, prepare_features, time_based_split


def test_parse_issue_date(sample_loans_df):
    df = parse_issue_date(sample_loans_df)
    assert "issue_date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["issue_date"])
    assert df["issue_date"].notna().all()


def test_time_based_split_no_overlap(sample_loans_df):
    train, test = time_based_split(sample_loans_df, cutoff_date="2015-01-01")
    assert train["issue_date"].max() < pd.Timestamp("2015-01-01")
    assert test["issue_date"].min() >= pd.Timestamp("2015-01-01")
    assert len(train) + len(test) == len(sample_loans_df)


def test_time_based_split_produces_both_sets(sample_loans_df):
    # Fixture spans 2007-2019ish, so a mid-range cutoff should produce both non-empty
    train, test = time_based_split(sample_loans_df, cutoff_date="2013-01-01")
    assert len(train) > 0
    assert len(test) > 0


def test_prepare_features_fills_missing_emp_length(sample_loans_df):
    assert sample_loans_df["emp_length_years"].isna().any()  # fixture has NaNs by design
    prepared = prepare_features(sample_loans_df)
    assert prepared["emp_length_years"].isna().sum() == 0


def test_prepare_features_raises_on_missing_column(sample_loans_df):
    broken = sample_loans_df.drop(columns=["dti"])
    with pytest.raises(ValueError, match="missing expected columns"):
        prepare_features(broken)
