"""
Tests for src/data_loader.py.

Uses a small, in-memory synthetic CSV fixture -- NOT the real Lending Club file --
so cleaning/parsing/filtering logic is verified deterministically and fast,
independent of whether the multi-GB real dataset is present on disk.
"""
import pandas as pd
import pytest

from src.data_loader import (
    IngestionError,
    RunLogger,
    _parse_emp_length,
    _parse_int_rate,
    _parse_term,
    clean_and_scope,
    load_raw,
)


@pytest.fixture
def raw_fixture(tmp_path):
    """A tiny synthetic dataset covering the edge cases the pipeline needs to handle."""
    data = pd.DataFrame({
        "loan_amnt": [10000, 5000, 8000, 12000, 3000, 15000, 9000],
        "term": [" 36 months", " 60 months", " 36 months", " 60 months", " 36 months", " 36 months", " 36 months"],
        "int_rate": ["13.56%", "18.25%", "11.00%", "22.10%", "9.50%", "14.00%", "16.00%"],
        "grade": ["C", "D", "A", "E", "B", "F", "D"],          # A and B should be filtered out
        "sub_grade": ["C3", "D1", "A2", "E4", "B1", "F2", "D2"],
        "emp_length": ["10+ years", "< 1 year", "5 years", "n/a", "2 years", "3 years", "4 years"],
        "annual_inc": [55000, 40000, 60000, 35000, 45000, 30000, 48000],
        "dti": [18.5, 22.0, 15.0, 28.0, 20.0, 25.0, 19.0],
        # "Current" and "Issued" should be filtered out (no known outcome yet)
        "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Default", "Current", "Fully Paid", "Fully Paid"],
        "purpose": ["debt_consolidation", "credit_card", "other", "medical", "debt_consolidation", "home_improvement", "credit_card"],
        "fico_range_low": [680, 660, 700, 620, 690, 610, 665],
        "fico_range_high": [684, 664, 704, 624, 694, 614, 669],
        # Last row is pre-2012 -- should be filtered out by the platform-maturity scope
        "issue_d": ["Jan-2018", "Mar-2018", "Feb-2018", "Apr-2018", "May-2018", "Jun-2018", "Jun-2009"],
        "revol_util": ["45.2%", "60.1%", "20.0%", "80.5%", "35.0%", "70.2%", "50.0%"],
        "delinq_2yrs": [0, 1, 0, 2, 0, 1, 0],
        "open_acc": [8, 5, 10, 4, 9, 6, 7],
        "pub_rec": [0, 0, 0, 1, 0, 0, 0],
        "inq_last_6mths": [1, 2, 0, 3, 1, 2, 1],
        "home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT", "MORTGAGE", "RENT", "OWN"],
        "verification_status": ["Verified", "Not Verified", "Source Verified", "Verified", "Not Verified", "Verified", "Verified"],
        "mort_acc": [1, 0, 2, 0, 1, 0, 1],
        "total_acc": [15, 8, 20, 6, 12, 9, 11],
        # last_pymnt_d after issue_d for all except one intentionally invalid row (Apr-2018 row)
        "last_pymnt_d": ["Dec-2019", "Sep-2018", "Nov-2020", "Feb-2018", "Aug-2018", "Dec-2020", "May-2009"],
    })
    path = tmp_path / "raw.csv"
    data.to_csv(path, index=False)
    return path


def test_parse_term():
    s = pd.Series([" 36 months", " 60 months"])
    result = _parse_term(s)
    assert result[0].tolist() == [36.0, 60.0]


def test_parse_int_rate_string():
    s = pd.Series(["13.56%", "9.00%"])
    result = _parse_int_rate(s)
    assert result.iloc[0] == pytest.approx(0.1356)
    assert result.iloc[1] == pytest.approx(0.09)


def test_parse_int_rate_numeric():
    s = pd.Series([13.56, 9.0])
    result = _parse_int_rate(s)
    assert result.iloc[0] == pytest.approx(0.1356)


def test_parse_emp_length():
    s = pd.Series(["10+ years", "< 1 year", "5 years", "n/a", None])
    result = _parse_emp_length(s)
    assert result.iloc[0] == 10.0
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == 5.0
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])


def test_load_raw_missing_file(tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    with pytest.raises(IngestionError, match="not found"):
        load_raw(tmp_path / "does_not_exist.csv", logger)


def test_load_raw_missing_columns(tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"only_one_column": [1, 2, 3]}).to_csv(bad_path, index=False)
    with pytest.raises(IngestionError, match="missing required columns"):
        load_raw(bad_path, logger)


def test_clean_and_scope_filters_grade(raw_fixture, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    df = load_raw(raw_fixture, logger)
    cleaned = clean_and_scope(df, logger)
    # Grades A and B should be gone; only C, D, E, F remain
    assert set(cleaned["grade"].unique()).issubset({"C", "D", "E", "F"})


def test_clean_and_scope_filters_loan_status(raw_fixture, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    df = load_raw(raw_fixture, logger)
    cleaned = clean_and_scope(df, logger)
    # "Current" status should be excluded (no known outcome yet)
    assert "Current" not in cleaned["loan_status"].values


def test_clean_and_scope_derives_default_label(raw_fixture, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    df = load_raw(raw_fixture, logger)
    cleaned = clean_and_scope(df, logger)
    fully_paid = cleaned[cleaned["loan_status"] == "Fully Paid"]
    charged_off_or_default = cleaned[cleaned["loan_status"].isin(["Charged Off", "Default"])]
    assert (fully_paid["defaulted"] == 0).all()
    assert (charged_off_or_default["defaulted"] == 1).all()


def test_clean_and_scope_filters_platform_maturity(raw_fixture, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    df = load_raw(raw_fixture, logger)
    cleaned = clean_and_scope(df, logger)
    # The Jun-2009 row should be excluded -- pre-2012 platform-maturity scope
    assert "Jun-2009" not in cleaned["issue_d"].values
    assert cleaned["issue_d"].str.contains("2009").sum() == 0


def test_clean_and_scope_computes_months_on_book(raw_fixture, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    df = load_raw(raw_fixture, logger)
    cleaned = clean_and_scope(df, logger)
    assert "months_on_book" in cleaned.columns
    assert (cleaned["months_on_book"] >= 0).all()
    # Row 0: issued Jan-2018, last payment Dec-2019 -> 23 months
    row0 = cleaned[cleaned["issue_d"] == "Jan-2018"].iloc[0]
    assert row0["months_on_book"] == 23


def test_clean_and_scope_filters_invalid_months_on_book(raw_fixture, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    df = load_raw(raw_fixture, logger)
    cleaned = clean_and_scope(df, logger)
    # The Apr-2018 row has last_pymnt_d of Feb-2018 (before issue_d) -- invalid, should be dropped
    assert "Apr-2018" not in cleaned["issue_d"].values


def test_clean_and_scope_computes_fico_avg(raw_fixture, tmp_path):
    logger = RunLogger(tmp_path / "log.jsonl")
    df = load_raw(raw_fixture, logger)
    cleaned = clean_and_scope(df, logger)
    row = cleaned.iloc[0]
    assert row["fico_avg"] == pytest.approx((row["fico_range_low"] + row["fico_range_high"]) / 2)


def test_run_logger_writes_jsonl(tmp_path):
    log_path = tmp_path / "log.jsonl"
    logger = RunLogger(log_path)
    logger.log("test_step", rows_in=100, rows_out=90)
    logger.flush()
    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    assert '"step": "test_step"' in lines[0]
