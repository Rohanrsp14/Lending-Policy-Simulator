"""
Vintage loss-emergence curve: for each loan origination cohort (vintage,
defined by issue year -- see CLAUDE.md for why time, not grade, is the right
cohort definition), shows cumulative default rate as a function of months on
book.

Built on REAL time-to-default data (months_on_book, from src/data_loader.py's
last_pymnt_d-derived field -- see PR 2.5), not a synthetic/illustrative
maturation curve. Denominator convention: cumulative default rate at month m
is (number of loans in the vintage that had defaulted by month m) / (total
loans in that vintage) -- a static cohort denominator, the standard "vintage
curve" convention in credit risk, not a survival-adjusted rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_issue_year(df: pd.DataFrame) -> pd.DataFrame:
    """Adds an issue_year column (int), derived from issue_d, if not already present."""
    df = df.copy()
    if "issue_year" not in df.columns:
        df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year
    return df


def compute_vintage_curve(df: pd.DataFrame, max_months: int = 60) -> pd.DataFrame:
    """
    Returns one row per (issue_year, month) with the cumulative default rate
    at that month, for months 0..max_months. Requires issue_year (or issue_d,
    which will be parsed), months_on_book, and defaulted columns.
    """
    df = add_issue_year(df)
    months_range = np.arange(0, max_months + 1)
    rows = []

    for vintage_year, vintage_df in df.groupby("issue_year"):
        total = len(vintage_df)
        if total == 0:
            continue
        defaulted_months = np.sort(
            vintage_df.loc[vintage_df["defaulted"] == 1, "months_on_book"].values
        )
        # searchsorted gives, for each month m, how many defaulted_months are <= m
        cum_counts = np.searchsorted(defaulted_months, months_range, side="right")
        cum_rates = cum_counts / total

        for m, rate, cnt in zip(months_range, cum_rates, cum_counts):
            rows.append({
                "issue_year": int(vintage_year),
                "month": int(m),
                "cumulative_default_rate": float(rate),
                "cumulative_defaults": int(cnt),
                "vintage_size": total,
            })

    return pd.DataFrame(rows)


def vintage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per vintage year: total loans, ultimate (final) default rate,
    and median months-to-default among loans that defaulted. A quick-read
    companion to the full month-by-month curve.
    """
    df = add_issue_year(df)
    summary = df.groupby("issue_year").agg(
        vintage_size=("defaulted", "size"),
        ultimate_default_rate=("defaulted", "mean"),
    ).reset_index()

    median_months = (
        df[df["defaulted"] == 1]
        .groupby("issue_year")["months_on_book"]
        .median()
        .rename("median_months_to_default")
    )
    summary = summary.merge(median_months, on="issue_year", how="left")
    return summary
