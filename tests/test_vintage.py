import numpy as np
import pandas as pd
import pytest

from src.vintage import add_issue_year, compute_vintage_curve, vintage_summary


@pytest.fixture
def vintage_fixture():
    """
    A small, hand-checkable dataset: 2 vintages, known default timing.

    2013 vintage (4 loans): 2 defaults at month 6 and month 12, 2 fully paid.
    2014 vintage (2 loans): 1 default at month 3, 1 fully paid.
    """
    return pd.DataFrame({
        "issue_d": ["Jan-2013", "Jan-2013", "Jan-2013", "Jan-2013", "Jun-2014", "Jun-2014"],
        "defaulted": [1, 1, 0, 0, 1, 0],
        "months_on_book": [6, 12, 36, 36, 3, 24],
    })


def test_add_issue_year(vintage_fixture):
    df = add_issue_year(vintage_fixture)
    assert list(df["issue_year"]) == [2013, 2013, 2013, 2013, 2014, 2014]


def test_compute_vintage_curve_zero_before_first_default(vintage_fixture):
    curve = compute_vintage_curve(vintage_fixture, max_months=36)
    v2013 = curve[curve["issue_year"] == 2013]
    # Before month 6, no defaults have occurred yet in the 2013 vintage
    assert (v2013[v2013["month"] < 6]["cumulative_default_rate"] == 0).all()


def test_compute_vintage_curve_matches_hand_calculation(vintage_fixture):
    curve = compute_vintage_curve(vintage_fixture, max_months=36)
    v2013 = curve[curve["issue_year"] == 2013].set_index("month")
    # At month 6: 1 of 4 defaulted so far -> 0.25
    assert v2013.loc[6, "cumulative_default_rate"] == pytest.approx(0.25)
    # At month 12: both defaults have occurred -> 2 of 4 = 0.5
    assert v2013.loc[12, "cumulative_default_rate"] == pytest.approx(0.5)
    # At month 36 (end): still 0.5, no further defaults occur
    assert v2013.loc[36, "cumulative_default_rate"] == pytest.approx(0.5)


def test_compute_vintage_curve_is_monotonic_nondecreasing(vintage_fixture):
    curve = compute_vintage_curve(vintage_fixture, max_months=36)
    for year, group in curve.groupby("issue_year"):
        rates = group.sort_values("month")["cumulative_default_rate"].values
        assert (np.diff(rates) >= 0).all()


def test_vintage_summary_ultimate_default_rate(vintage_fixture):
    summary = vintage_summary(vintage_fixture)
    row_2013 = summary[summary["issue_year"] == 2013].iloc[0]
    row_2014 = summary[summary["issue_year"] == 2014].iloc[0]
    assert row_2013["ultimate_default_rate"] == pytest.approx(0.5)   # 2 of 4
    assert row_2014["ultimate_default_rate"] == pytest.approx(0.5)   # 1 of 2
    assert row_2013["vintage_size"] == 4
    assert row_2014["vintage_size"] == 2


def test_vintage_summary_median_months_to_default(vintage_fixture):
    summary = vintage_summary(vintage_fixture)
    row_2013 = summary[summary["issue_year"] == 2013].iloc[0]
    # Defaults at month 6 and 12 -> median = 9
    assert row_2013["median_months_to_default"] == pytest.approx(9)
