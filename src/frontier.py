"""
Approval/return frontier: sweeps a range of target approval rates and shows,
for EACH policy, the resulting approval rate vs. RAROC tradeoff at that point
-- not just challenger, symmetrically for champion too. This fixes an
asymmetry flagged after PR 2.4: champion had only ever been evaluated at a
single calibrated cutoff, while challenger got a full RAROC-optimization
sweep. A genuine "frontier" needs both policies to explore their full
tradeoff space, not just one. See CLAUDE.md.

Cutoffs are calibrated on TRAIN only (no leakage) and applied to TEST for
every point on the frontier, reusing the same calibration functions already
built in src/models.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models import (
    CAPITAL_RATE,
    LGD,
    OPEX_RATE,
    calibrate_fico_cutoff,
    calibrate_pd_threshold,
    champion_decision,
    challenger_decision,
    compute_raroc,
)


def compute_frontier(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame,
                      quantiles: np.ndarray | None = None,
                      lgd: float = LGD, opex_rate: float = OPEX_RATE,
                      capital_rate: float = CAPITAL_RATE) -> pd.DataFrame:
    """
    Returns one row per target approval-rate quantile, with champion's and
    challenger's cutoff (calibrated on train), and both policies' resulting
    approval rate and RAROC when applied to test. Plot champion_approval_rate
    vs. champion_raroc and challenger_approval_rate vs. challenger_raroc as
    two curves on the same axes for the full frontier comparison.

    lgd/opex_rate/capital_rate default to the illustrative module constants
    but can be overridden -- this is what lets src/sensitivity.py stress-test
    whether champion's RAROC advantage over challenger holds across a
    plausible range of assumptions, not just the one fixed set used
    throughout PRs 2-3.1. Note: cutoff CALIBRATION (which score threshold
    hits a target approval rate) does not depend on these economic
    parameters -- only the resulting RAROC does -- so calibrate_fico_cutoff
    and calibrate_pd_threshold are unaffected and don't need these arguments.
    """
    if quantiles is None:
        quantiles = np.linspace(0.50, 0.99, 25)

    rows = []
    for q in quantiles:
        fico_cutoff = calibrate_fico_cutoff(train_df, q)
        pd_threshold = calibrate_pd_threshold(model, train_df, q)

        champ_mask = champion_decision(test_df, cutoff=fico_cutoff)
        chall_mask = challenger_decision(model, test_df, pd_threshold)

        champ_metrics = compute_raroc(test_df, champ_mask, lgd=lgd, opex_rate=opex_rate, capital_rate=capital_rate)
        chall_metrics = compute_raroc(test_df, chall_mask, lgd=lgd, opex_rate=opex_rate, capital_rate=capital_rate)

        rows.append({
            "target_quantile": q,
            "champion_fico_cutoff": fico_cutoff,
            "champion_approval_rate": champ_metrics["approval_rate"],
            "champion_raroc": champ_metrics["raroc"],
            "champion_loss_rate": champ_metrics["loss_rate"],
            "challenger_pd_threshold": pd_threshold,
            "challenger_approval_rate": chall_metrics["approval_rate"],
            "challenger_raroc": chall_metrics["raroc"],
            "challenger_loss_rate": chall_metrics["loss_rate"],
        })

    return pd.DataFrame(rows)


def best_point(frontier_df: pd.DataFrame, policy: str) -> pd.Series:
    """
    Returns the row with the highest RAROC for the given policy ('champion'
    or 'challenger') -- the operating point that policy's own frontier says
    is best, for direct comparison against the other policy's best point.
    """
    if policy not in ("champion", "challenger"):
        raise ValueError("policy must be 'champion' or 'challenger'")
    raroc_col = f"{policy}_raroc"
    return frontier_df.loc[frontier_df[raroc_col].idxmax()]
