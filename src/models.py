"""
Champion vs. challenger policy definitions for the Lending Policy Simulator.

Champion: the current, rule-based underwriting cutoff (FICO threshold) -- no
model, exactly how a Regional Finance-tier lender plausibly still approves
loans today.

Challenger: a trained logistic-regression PD (probability-of-default) model,
proposed as a statistical improvement over the rule-based cutoff.

RAROC for both policies is computed from REALIZED historical outcomes on a
held-out (time-based) test set, not from a model's own predicted probability
-- this is a deliberate choice: it makes the two policies directly comparable
on real, known-outcome data rather than trusting either policy's own risk
estimate for the loss side of the ledger. See CLAUDE.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

# ---- Portfolio economics assumptions (documented, not hidden) ----------
# These are simplifying assumptions for an illustrative RAROC calculation,
# not calibrated to any real institution's actual economics. See CLAUDE.md
# and README.md.
LGD = 0.55          # Loss given default
OPEX_RATE = 0.05     # Operating cost, as a fraction of approved loan amount
CAPITAL_RATE = 0.08  # Regulatory-style capital charge, as a fraction of approved loan amount


def amortized_interest(loan_amnt: np.ndarray, int_rate_frac: np.ndarray, term_months: np.ndarray) -> np.ndarray:
    """
    Total interest income over the full loan term, using the standard
    fixed-payment installment amortization formula -- replaces an earlier,
    cruder flat multiplier (loan_amnt * rate * a fudge factor).

    KNOWN LIMITATION (documented, not hidden): this assumes every approved
    loan runs to full term. Loans that default early stop paying before
    then, so this OVERSTATES revenue on defaulted loans specifically. A
    fully correct cash-flow model would need each loan's actual time-to-
    default, which is exactly what the vintage/time-on-book analysis in
    PR 3 adds -- this fix is a real improvement over the prior flat
    multiplier, but the early-default revenue truncation is intentionally
    deferred to that PR, not silently ignored. See CLAUDE.md.
    """
    monthly_rate = int_rate_frac / 12
    n = term_months
    payment = loan_amnt * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)
    total_interest = payment * n - loan_amnt
    return total_interest

# Champion cutoff: a stated, documented assumption representing a plausible
# current underwriting practice within the C-F grade population. Because this
# dataset only contains ALREADY-ACCEPTED loans (Lending Club does not publish
# rejected applications), a fixed FICO cutoff can turn out to be non-binding --
# i.e. everyone in the scoped population already clears it, since they were
# already screened by Lending Club's own real underwriting before this data
# was ever published. This is a known limitation of "accepted loans only"
# data called reject inference -- see CLAUDE.md and README.md.
#
# To get a meaningful, binding comparison despite this, champion and
# challenger are calibrated to the SAME approval rate (a "swap set" analysis:
# same volume, different mix) rather than relying on an arbitrary absolute
# FICO number that may not bind on this population.
CHAMPION_FICO_CUTOFF = 660  # kept as a documented reference constant; see calibrate_fico_cutoff for the binding version
DEFAULT_TARGET_APPROVAL_RATE = 0.85


def champion_decision(df: pd.DataFrame, cutoff: float = CHAMPION_FICO_CUTOFF) -> pd.Series:
    """Rule-based approval: approve if fico_avg >= cutoff. No model involved."""
    return df["fico_avg"] >= cutoff


def calibrate_fico_cutoff(train_df: pd.DataFrame, target_approval_rate: float = DEFAULT_TARGET_APPROVAL_RATE) -> float:
    """
    Finds the FICO cutoff, calibrated on the TRAINING population only (avoids
    leakage), that would approve approximately target_approval_rate of loans --
    e.g. 0.85 means "decline the riskiest 15% by raw FICO". This makes champion
    a genuinely binding policy regardless of where this specific population's
    FICO distribution happens to sit.
    """
    decline_rate = 1 - target_approval_rate
    return float(train_df["fico_avg"].quantile(decline_rate))


def champion_decision_volume_matched(df: pd.DataFrame, train_df: pd.DataFrame,
                                       target_approval_rate: float = DEFAULT_TARGET_APPROVAL_RATE) -> pd.Series:
    """Champion, calibrated to approve target_approval_rate of the population (e.g. 85%)."""
    cutoff = calibrate_fico_cutoff(train_df, target_approval_rate)
    return df["fico_avg"] >= cutoff


def build_challenger_pipeline() -> Pipeline:
    """Logistic regression PD model: numeric features scaled, purpose one-hot encoded."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def train_challenger(train_df: pd.DataFrame) -> Pipeline:
    """Fits the challenger PD model on the training split only."""
    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train_df[TARGET]
    pipeline = build_challenger_pipeline()
    pipeline.fit(X, y)
    return pipeline


def challenger_decision(model: Pipeline, df: pd.DataFrame, pd_threshold: float) -> pd.Series:
    """Approve if the model's predicted default probability is at or below the threshold."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predicted_pd = model.predict_proba(X)[:, 1]
    return pd.Series(predicted_pd <= pd_threshold, index=df.index)


def calibrate_pd_threshold(model: Pipeline, train_df: pd.DataFrame,
                            target_approval_rate: float = DEFAULT_TARGET_APPROVAL_RATE) -> float:
    """
    Finds the PD threshold, calibrated on the TRAINING population's predicted
    probabilities, that would approve approximately target_approval_rate of
    loans -- the challenger's equivalent of calibrate_fico_cutoff, so champion
    and challenger can be compared at the same approval volume (a swap-set
    analysis: same volume, different mix of who gets approved).
    """
    train_pd = predict_pd(model, train_df)
    return float(np.quantile(train_pd, target_approval_rate))


def challenger_decision_volume_matched(model: Pipeline, df: pd.DataFrame, train_df: pd.DataFrame,
                                         target_approval_rate: float = DEFAULT_TARGET_APPROVAL_RATE) -> pd.Series:
    """Challenger, calibrated to approve target_approval_rate of the population (e.g. 85%)."""
    threshold = calibrate_pd_threshold(model, train_df, target_approval_rate)
    return challenger_decision(model, df, threshold)


def sweep_pd_thresholds(model: Pipeline, train_df: pd.DataFrame, quantiles: np.ndarray | None = None) -> pd.DataFrame:
    """
    Evaluates RAROC (and other metrics) across a range of PD thresholds on the
    TRAINING set only (avoids leakage -- the test set is never used to pick a
    threshold). Returns one row per candidate threshold. This is the
    volume-matching approach's counterpart optimized for economics instead of
    a fixed approval rate -- and it doubles as groundwork for PR 3's
    approval/return frontier, which is the same sweep-and-plot idea applied
    to champion's FICO cutoff instead of challenger's PD threshold.
    """
    if quantiles is None:
        quantiles = np.linspace(0.05, 0.95, 19)

    train_pd = predict_pd(model, train_df)
    candidate_thresholds = np.quantile(train_pd, quantiles)

    rows = []
    for q, threshold in zip(quantiles, candidate_thresholds):
        mask = challenger_decision(model, train_df, threshold)
        metrics = compute_raroc(train_df, mask)
        rows.append({"target_quantile": q, "pd_threshold": threshold, **metrics})

    return pd.DataFrame(rows)


def calibrate_pd_threshold_for_raroc(model: Pipeline, train_df: pd.DataFrame,
                                       quantiles: np.ndarray | None = None) -> float:
    """
    Sweeps PD thresholds on the TRAINING set and returns the one that
    maximizes RAROC directly -- rather than calibrate_pd_threshold's approach
    of matching a target approval volume. Reflects the real objective (risk-
    adjusted return), not a proxy (approval rate) -- see CLAUDE.md for why
    volume-matching alone was found to be an incomplete comparison.
    """
    sweep = sweep_pd_thresholds(model, train_df, quantiles)
    best_row = sweep.loc[sweep["raroc"].idxmax()]
    return float(best_row["pd_threshold"])


def challenger_decision_raroc_optimized(model: Pipeline, df: pd.DataFrame, train_df: pd.DataFrame,
                                          quantiles: np.ndarray | None = None) -> pd.Series:
    """Challenger, calibrated on train to directly maximize RAROC rather than match a volume target."""
    threshold = calibrate_pd_threshold_for_raroc(model, train_df, quantiles)
    return challenger_decision(model, df, threshold)


def predict_pd(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return model.predict_proba(X)[:, 1]


def compute_raroc(df: pd.DataFrame, approved_mask: pd.Series) -> dict:
    """
    Computes the P&L waterfall to RAROC for an approved population, using
    REAL loan_amnt/int_rate_frac and REALIZED (actual, historical) default
    outcomes -- not a model-predicted probability. This is what makes
    champion and challenger directly comparable on the same held-out data.

    IMPORTANT -- annualized, not lifetime-total: amortized_interest() and the
    realized loss are both full-loan-life figures (e.g. 3-5 years' worth).
    capital and opex are point-in-time charges on the loan amount, meant to
    represent an ANNUAL charge (standard RAROC convention). Comparing a
    multi-year revenue total directly against a one-year capital base
    produces a nonsensical RAROC (a real bug caught by running this against
    real data -- see CLAUDE.md). Revenue and loss are therefore divided by
    each loan's own term-in-years before being summed, so every quantity
    feeding RAROC is on the same annual basis as the capital charge.
    """
    approved = df[approved_mask]
    n = len(approved)
    if n == 0:
        return {
            "n": 0, "approval_rate": 0.0, "revenue": 0.0, "actual_loss": 0.0,
            "opex": 0.0, "capital": 0.0, "net_income": 0.0, "raroc": 0.0,
            "loss_rate": 0.0,
        }

    term_years = approved["term_months"].values / 12

    lifetime_interest = amortized_interest(
        approved["loan_amnt"].values,
        approved["int_rate_frac"].values,
        approved["term_months"].values,
    )
    annual_revenue = (lifetime_interest / term_years).sum()

    lifetime_loss = (approved["loan_amnt"] * approved[TARGET] * LGD).values
    annual_loss = (lifetime_loss / term_years).sum()

    opex = (approved["loan_amnt"] * OPEX_RATE).sum()
    capital = (approved["loan_amnt"] * CAPITAL_RATE).sum()
    net_income = annual_revenue - annual_loss - opex
    raroc = net_income / capital if capital > 0 else 0.0
    loss_rate = lifetime_loss.sum() / approved["loan_amnt"].sum()  # unannualized -- a simple lifetime loss rate, easier to read

    return {
        "n": n,
        "approval_rate": n / len(df),
        "revenue": annual_revenue,
        "actual_loss": annual_loss,
        "opex": opex,
        "capital": capital,
        "net_income": net_income,
        "raroc": raroc,
        "loss_rate": loss_rate,
    }


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the good/bad cumulative distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate_challenger(model: Pipeline, test_df: pd.DataFrame) -> dict:
    """Standard ranking-quality metrics for the challenger PD model on held-out data."""
    y_true = test_df[TARGET].values
    y_score = predict_pd(model, test_df)
    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1
    ks = ks_statistic(y_true, y_score)
    return {"auc": auc, "gini": gini, "ks": ks}
