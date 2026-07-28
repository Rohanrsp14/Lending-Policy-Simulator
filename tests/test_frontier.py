import pytest

from src.features import parse_issue_date, prepare_features, time_based_split
from src.frontier import best_point, compute_frontier
from src.models import train_challenger


@pytest.fixture
def prepared_split_for_frontier(sample_loans_df):
    df = prepare_features(sample_loans_df)
    df = parse_issue_date(df)
    train, test = time_based_split(df, cutoff_date="2015-06-01")
    return train, test


def test_compute_frontier_returns_expected_columns(prepared_split_for_frontier):
    train, test = prepared_split_for_frontier
    model = train_challenger(train)
    frontier = compute_frontier(model, train, test)
    expected_cols = {
        "target_quantile", "champion_fico_cutoff", "champion_approval_rate",
        "champion_raroc", "challenger_pd_threshold", "challenger_approval_rate",
        "challenger_raroc",
    }
    assert expected_cols.issubset(frontier.columns)


def test_compute_frontier_approval_rate_increases_with_quantile(prepared_split_for_frontier):
    train, test = prepared_split_for_frontier
    model = train_challenger(train)
    frontier = compute_frontier(model, train, test).sort_values("target_quantile")
    # Higher target quantile -> higher (or equal) approval rate for BOTH policies,
    # since both are calibrated as "approve the top X% by score" on train.
    champ_rates = frontier["champion_approval_rate"].values
    chall_rates = frontier["challenger_approval_rate"].values
    assert (champ_rates[-1] >= champ_rates[0])
    assert (chall_rates[-1] >= chall_rates[0])


def test_compute_frontier_is_symmetric_for_both_policies(prepared_split_for_frontier):
    """Both champion and challenger must be swept, not just challenger -- the
    asymmetry this PR was built to fix."""
    train, test = prepared_split_for_frontier
    model = train_challenger(train)
    frontier = compute_frontier(model, train, test)
    assert frontier["champion_fico_cutoff"].nunique() > 1
    assert frontier["challenger_pd_threshold"].nunique() > 1


def test_best_point_selects_max_raroc_row(prepared_split_for_frontier):
    train, test = prepared_split_for_frontier
    model = train_challenger(train)
    frontier = compute_frontier(model, train, test)
    champ_best = best_point(frontier, "champion")
    assert champ_best["champion_raroc"] == frontier["champion_raroc"].max()


def test_best_point_raises_on_invalid_policy(prepared_split_for_frontier):
    train, test = prepared_split_for_frontier
    model = train_challenger(train)
    frontier = compute_frontier(model, train, test)
    with pytest.raises(ValueError, match="policy must be"):
        best_point(frontier, "referee")
