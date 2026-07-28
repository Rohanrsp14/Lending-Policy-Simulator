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
AVG_LIFE = 1.4       # Revenue multiplier approximating loan term/renewal effects

# Champion cutoff: a stated assumption representing a plausible current
# underwriting practice within the C-F grade population, NOT derived from
# any real institution's actual policy. Flagged as an ask-first item in
# CLAUDE.md if this needs to change.
CHAMPION_FICO_CUTOFF = 660


def champion_decision(df: pd.DataFrame, cutoff: float = CHAMPION_FICO_CUTOFF) -> pd.Series:
    """Rule-based approval: approve if fico_avg >= cutoff. No model involved."""
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


def predict_pd(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return model.predict_proba(X)[:, 1]


def compute_raroc(df: pd.DataFrame, approved_mask: pd.Series) -> dict:
    """
    Computes the P&L waterfall to RAROC for an approved population, using
    REAL loan_amnt/int_rate_frac and REALIZED (actual, historical) default
    outcomes -- not a model-predicted probability. This is what makes
    champion and challenger directly comparable on the same held-out data.
    """
    approved = df[approved_mask]
    n = len(approved)
    if n == 0:
        return {
            "n": 0, "approval_rate": 0.0, "revenue": 0.0, "actual_loss": 0.0,
            "opex": 0.0, "capital": 0.0, "net_income": 0.0, "raroc": 0.0,
            "loss_rate": 0.0,
        }

    revenue = (approved["loan_amnt"] * approved["int_rate_frac"] * AVG_LIFE).sum()
    actual_loss = (approved["loan_amnt"] * approved[TARGET] * LGD).sum()
    opex = (approved["loan_amnt"] * OPEX_RATE).sum()
    capital = (approved["loan_amnt"] * CAPITAL_RATE).sum()
    net_income = revenue - actual_loss - opex
    raroc = net_income / capital if capital > 0 else 0.0
    loss_rate = actual_loss / approved["loan_amnt"].sum()

    return {
        "n": n,
        "approval_rate": n / len(df),
        "revenue": revenue,
        "actual_loss": actual_loss,
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
