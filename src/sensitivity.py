"""
RAROC sensitivity analysis: stress-tests the champion-vs-challenger
conclusion (docs/MODEL_VALIDATION.md Section 6) across a plausible range of
the illustrative economic assumptions (LGD, opex rate, capital rate), one at
a time, holding the other two at their base-case value.

Purpose: every RAROC number in this project depends on three constants that
are explicitly documented as illustrative, not calibrated to any real
institution (src/models.py, CLAUDE.md). Rather than presenting a single
number as if it were certain, this module asks: does champion's RAROC
advantage over challenger hold across a realistic range of what those
assumptions could actually be? If yes, that's a genuinely more defensible
conclusion. If the conclusion flips somewhere in a plausible range, that's
equally important to know and say -- it tells a real business exactly how
much the recommendation depends on numbers only Treasury/Finance can supply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.frontier import best_point, compute_frontier
from src.models import CAPITAL_RATE, LGD, OPEX_RATE

DEFAULT_LGD_RANGE = np.linspace(0.35, 0.75, 5)
DEFAULT_OPEX_RANGE = np.linspace(0.02, 0.10, 5)
DEFAULT_CAPITAL_RANGE = np.linspace(0.04, 0.12, 5)


def _one_parameter_sweep(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame,
                          param_name: str, param_values: np.ndarray,
                          quantiles: np.ndarray | None = None) -> pd.DataFrame:
    """
    Sweeps ONE economic parameter across param_values, holding the other two
    at their base-case (module-default) value. For each value, finds each
    policy's own best point on ITS OWN frontier (re-optimized under the new
    assumption, not just re-evaluated at the original best point) and
    records the resulting RAROC and the delta between policies.

    quantiles forwards to compute_frontier's own quantiles argument -- pass
    a finer grid (e.g. np.linspace(0.50, 0.99, 100) instead of the default
    25 points) to confirm a found optimum is a genuine peak and not a
    coarse-grid artifact. See CLAUDE.md.
    """
    base = {"lgd": LGD, "opex_rate": OPEX_RATE, "capital_rate": CAPITAL_RATE}
    rows = []

    for value in param_values:
        kwargs = dict(base)
        kwargs[param_name] = value

        frontier = compute_frontier(model, train_df, test_df, quantiles=quantiles, **kwargs)
        champ_best = best_point(frontier, "champion")
        chall_best = best_point(frontier, "challenger")

        rows.append({
            "swept_parameter": param_name,
            "parameter_value": value,
            "champion_best_raroc": champ_best["champion_raroc"],
            "champion_best_approval_rate": champ_best["champion_approval_rate"],
            "challenger_best_raroc": chall_best["challenger_raroc"],
            "challenger_best_approval_rate": chall_best["challenger_approval_rate"],
            "delta_challenger_minus_champion": chall_best["challenger_raroc"] - champ_best["champion_raroc"],
            "champion_wins": champ_best["champion_raroc"] >= chall_best["challenger_raroc"],
        })

    return pd.DataFrame(rows)


def run_sensitivity(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame,
                     lgd_range: np.ndarray | None = None,
                     opex_range: np.ndarray | None = None,
                     capital_range: np.ndarray | None = None,
                     quantiles: np.ndarray | None = None) -> pd.DataFrame:
    """
    Runs a one-at-a-time sensitivity sweep across LGD, opex rate, and capital
    rate, holding the other two at base case for each sweep. Returns a single
    combined dataframe (filter by swept_parameter to isolate one sweep).

    quantiles: forwards to compute_frontier's grid granularity for every
    sweep point (default: compute_frontier's own default, 25 points from
    0.50 to 0.99). Pass a finer grid to validate that a found optimum
    approval rate is a genuine peak, not a coarse-grid artifact.
    """
    lgd_range = DEFAULT_LGD_RANGE if lgd_range is None else lgd_range
    opex_range = DEFAULT_OPEX_RANGE if opex_range is None else opex_range
    capital_range = DEFAULT_CAPITAL_RANGE if capital_range is None else capital_range

    results = [
        _one_parameter_sweep(model, train_df, test_df, "lgd", lgd_range, quantiles=quantiles),
        _one_parameter_sweep(model, train_df, test_df, "opex_rate", opex_range, quantiles=quantiles),
        _one_parameter_sweep(model, train_df, test_df, "capital_rate", capital_range, quantiles=quantiles),
    ]
    return pd.concat(results, ignore_index=True)


def summarize_robustness(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per swept parameter: does champion win at EVERY tested value
    (fully robust), NO tested value (challenger always wins), or SOME
    (the conclusion depends on the assumption -- the most actionable case
    to flag to a real business, since it tells them exactly which number
    they need to nail down before trusting either conclusion).
    """
    def classify(group):
        if group["champion_wins"].all():
            return "champion_always_wins"
        elif not group["champion_wins"].any():
            return "challenger_always_wins"
        else:
            return "conclusion_depends_on_assumption"

    rows = []
    for param, group in sensitivity_df.groupby("swept_parameter"):
        rows.append({
            "swept_parameter": param,
            "robustness": classify(group),
            "min_delta": group["delta_challenger_minus_champion"].min(),
            "max_delta": group["delta_challenger_minus_champion"].max(),
        })
    return pd.DataFrame(rows)
