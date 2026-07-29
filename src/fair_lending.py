"""
Fair-lending parity screen.

Lending Club's data has no protected-class field and no borrower name, so a
real BISG (Bayesian Improved Surname Geocoding -- the actual technique used
by regulators and lenders to proxy race/ethnicity when it isn't collected at
origination) cannot be run here: BISG normally combines a surname-based
probability AND a geography-based probability via Bayes' rule. With no name
field at all, only the geography leg is implementable.

**This module is explicitly illustrative, not evidentiary.** The state-level
probability table below is FABRICATED for demonstration purposes -- it is
NOT derived from real Census Bureau race/ethnicity-by-geography data. Using
it to draw any conclusion about real disparate impact, on this or any real
population, would be a misuse of this tool. See CLAUDE.md and
docs/MODEL_VALIDATION.md.

What this DOES demonstrate correctly: the actual mechanics of a four-fifths
parity screen, and the single-leg version of BISG's Bayesian proxy-assignment
logic -- useful for showing HOW the check works, not WHAT it would find on
real applicant demographics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# FABRICATED, illustrative-only state-level probabilities of the synthetic
# "Group B" proxy label. NOT real Census demographic data. Any state not
# listed falls back to DEFAULT_STATE_PROBABILITY.
ILLUSTRATIVE_STATE_PROBABILITY = {
    "CA": 0.35, "TX": 0.30, "NY": 0.32, "FL": 0.28, "IL": 0.27,
    "GA": 0.33, "NC": 0.26, "OH": 0.20, "PA": 0.21, "MI": 0.22,
}
DEFAULT_STATE_PROBABILITY = 0.20

FOUR_FIFTHS_THRESHOLD = 0.80


def assign_geography_proxy(df: pd.DataFrame, seed: int = 20260101) -> pd.DataFrame:
    """
    Assigns a synthetic 'protected_class_proxy' (Group A / Group B) per loan,
    using addr_state to look up a FABRICATED, illustrative probability, then
    drawing group membership as a Bernoulli trial from that probability --
    the same Bayesian-draw mechanism BISG uses, applied to only the
    geography leg (no name field exists in this dataset for a surname leg).

    Adds two columns: geo_proxy_probability (the illustrative per-state
    probability used) and protected_class_proxy ('Group A' or 'Group B').
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    df["geo_proxy_probability"] = df["addr_state"].map(
        lambda s: ILLUSTRATIVE_STATE_PROBABILITY.get(s, DEFAULT_STATE_PROBABILITY)
    )
    draws = rng.random(len(df))
    df["protected_class_proxy"] = np.where(
        draws < df["geo_proxy_probability"], "Group B", "Group A"
    )
    return df


def four_fifths_ratio(rate_a: float, rate_b: float) -> float:
    """Standard four-fifths rule ratio: min(rate) / max(rate). 1.0 = perfect parity."""
    if rate_a == 0 and rate_b == 0:
        return 1.0
    return min(rate_a, rate_b) / max(rate_a, rate_b)


def parity_for_policy(df: pd.DataFrame, approved_mask: pd.Series) -> dict:
    """
    Approval rate for each proxy group, and the four-fifths ratio, for a
    single policy's approval mask.
    """
    group_a = df["protected_class_proxy"] == "Group A"
    group_b = df["protected_class_proxy"] == "Group B"

    rate_a = approved_mask[group_a].mean() if group_a.sum() > 0 else 0.0
    rate_b = approved_mask[group_b].mean() if group_b.sum() > 0 else 0.0
    ratio = four_fifths_ratio(rate_a, rate_b)

    return {
        "group_a_approval_rate": rate_a,
        "group_b_approval_rate": rate_b,
        "four_fifths_ratio": ratio,
        "flagged": ratio < FOUR_FIFTHS_THRESHOLD,
        "group_a_n": int(group_a.sum()),
        "group_b_n": int(group_b.sum()),
    }


def run_parity_screen(df: pd.DataFrame, policy_masks: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Runs the parity screen for each named policy (e.g. {'champion': mask1,
    'challenger_volume_matched': mask2, 'challenger_raroc_optimized': mask3}).
    Returns one row per policy.
    """
    rows = []
    for policy_name, mask in policy_masks.items():
        result = parity_for_policy(df, mask)
        rows.append({"policy": policy_name, **result})
    return pd.DataFrame(rows)
