import numpy as np
import pandas as pd
import pytest

from src.fair_lending import (
    FOUR_FIFTHS_THRESHOLD,
    assign_geography_proxy,
    four_fifths_ratio,
    parity_for_policy,
    run_parity_screen,
)


@pytest.fixture
def state_fixture():
    return pd.DataFrame({
        "addr_state": ["CA", "TX", "NY", "OH", "ZZ"] * 200,  # ZZ = unknown state, tests fallback
    })


def test_assign_geography_proxy_adds_expected_columns(state_fixture):
    df = assign_geography_proxy(state_fixture)
    assert "protected_class_proxy" in df.columns
    assert "geo_proxy_probability" in df.columns
    assert set(df["protected_class_proxy"].unique()).issubset({"Group A", "Group B"})


def test_assign_geography_proxy_uses_fallback_for_unknown_state(state_fixture):
    df = assign_geography_proxy(state_fixture)
    unknown_state_rows = df[df["addr_state"] == "ZZ"]
    assert (unknown_state_rows["geo_proxy_probability"] == 0.20).all()


def test_assign_geography_proxy_is_deterministic_with_same_seed(state_fixture):
    df1 = assign_geography_proxy(state_fixture, seed=42)
    df2 = assign_geography_proxy(state_fixture, seed=42)
    assert (df1["protected_class_proxy"] == df2["protected_class_proxy"]).all()


def test_assign_geography_proxy_differs_with_different_seed(state_fixture):
    df1 = assign_geography_proxy(state_fixture, seed=1)
    df2 = assign_geography_proxy(state_fixture, seed=2)
    # Not guaranteed to differ on every row, but should differ somewhere given 1000 rows
    assert not (df1["protected_class_proxy"] == df2["protected_class_proxy"]).all()


def test_four_fifths_ratio_perfect_parity():
    assert four_fifths_ratio(0.5, 0.5) == pytest.approx(1.0)


def test_four_fifths_ratio_order_independent():
    assert four_fifths_ratio(0.4, 0.8) == four_fifths_ratio(0.8, 0.4)


def test_four_fifths_ratio_both_zero_returns_one():
    assert four_fifths_ratio(0.0, 0.0) == 1.0


def test_four_fifths_ratio_flags_below_threshold():
    ratio = four_fifths_ratio(0.5, 0.9)  # 0.556, below 0.80
    assert ratio < FOUR_FIFTHS_THRESHOLD


def test_parity_for_policy_structure(state_fixture):
    df = assign_geography_proxy(state_fixture)
    approved_mask = pd.Series([True] * 500 + [False] * 500, index=df.index)
    result = parity_for_policy(df, approved_mask)
    expected_keys = {
        "group_a_approval_rate", "group_b_approval_rate", "four_fifths_ratio",
        "flagged", "group_a_n", "group_b_n",
    }
    assert set(result.keys()) == expected_keys
    assert 0.0 <= result["group_a_approval_rate"] <= 1.0
    assert 0.0 <= result["group_b_approval_rate"] <= 1.0


def test_run_parity_screen_returns_one_row_per_policy(state_fixture):
    df = assign_geography_proxy(state_fixture)
    n = len(df)
    masks = {
        "champion": pd.Series(np.arange(n) % 2 == 0, index=df.index),
        "challenger": pd.Series(np.arange(n) % 3 == 0, index=df.index),
    }
    result = run_parity_screen(df, masks)
    assert len(result) == 2
    assert set(result["policy"]) == {"champion", "challenger"}
