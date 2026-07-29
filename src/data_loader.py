"""
Data ingestion for the Lending Policy Simulator.

Loads the raw Lending Club accepted-loans export, scopes it to grades C-F
(near-prime/subprime, proxy for the Regional Finance lending tier) and to
loans with a matured, known outcome only, cleans/parses key fields, derives
a binary default label, and writes a cleaned Parquet file plus a structured
JSONL run log.

See CLAUDE.md for full data-provenance and scoping rules.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---- Config -----------------------------------------------------------

RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw/accepted_2007_to_2018Q4.csv"))
PROCESSED_DATA_PATH = Path("data/processed/loans_clean.parquet")
LOG_PATH = Path("logs/ingestion_runs.jsonl")

# Scope: near-prime/subprime grades only (proxy for Regional Finance tier)
SCOPED_GRADES = {"C", "D", "E", "F"}

# Only loans with a matured, known outcome are usable for loss-rate work.
# Current/Issued/Late statuses have no known final outcome yet and would
# bias loss rates if included (survivorship bias) -- see CLAUDE.md.
GOOD_STATUSES = {"Fully Paid"}
BAD_STATUSES = {"Charged Off", "Default"}
KEEP_STATUSES = GOOD_STATUSES | BAD_STATUSES

REQUIRED_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "annual_inc",
    "dti",
    "loan_status",
    "purpose",
    "fico_range_low",
    "fico_range_high",
    "issue_d",
    "revol_util",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "inq_last_6mths",
    "home_ownership",
    "verification_status",
    "mort_acc",
    "total_acc",
    "last_pymnt_d",
    "addr_state",
]

# Lending Club's earliest years (2007-2011) were a small, immature platform
# with materially different underwriting practices and origination volume
# than the post-2012 period. Mixing these regimes into one model adds noise
# and is a known contributor to modest AUC. Scoped out as a stated, locked
# assumption -- see CLAUDE.md.
MIN_ISSUE_YEAR = 2012


class IngestionError(Exception):
    """Raised when the raw file is missing required columns or is otherwise unusable."""


class RunLogger:
    """Minimal structured logger -- one JSON line per ingestion run, appended."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []

    def log(self, step: str, **fields):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            **fields,
        }
        self.events.append(event)

    def flush(self):
        with self.log_path.open("a") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")
        self.events = []


def _parse_term(term: pd.Series) -> pd.Series:
    """' 36 months' -> 36 (int)."""
    return term.astype(str).str.extract(r"(\d+)").astype(float)


def _parse_int_rate(int_rate: pd.Series) -> pd.Series:
    """'13.56%' -> 0.1356 (float). Handles already-numeric input too."""
    if pd.api.types.is_numeric_dtype(int_rate):
        # Already numeric -- assume it's a percentage like 13.56, convert to fraction.
        return int_rate.astype(float) / 100.0
    cleaned = int_rate.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce") / 100.0


def _parse_revol_util(revol_util: pd.Series) -> pd.Series:
    """'45.2%' -> 0.452 (float). Same format as int_rate; kept separate for clarity."""
    return _parse_int_rate(revol_util)


def _parse_issue_year(issue_d: pd.Series) -> pd.Series:
    """'Jan-2018' -> 2018 (int). Used only for the platform-maturity scope filter."""
    return pd.to_datetime(issue_d, format="%b-%Y", errors="coerce").dt.year


def _compute_months_on_book(issue_d: pd.Series, last_pymnt_d: pd.Series) -> pd.Series:
    """
    Months between issue_d and last_pymnt_d -- a real, data-derived proxy for
    time-to-terminal-event. For Charged Off/Default loans this approximates
    time-to-default; for Fully Paid loans it's time-to-payoff (which can be
    less than term_months if the loan was prepaid early). This replaces an
    earlier design that would have needed a synthetic/illustrative
    maturation curve -- see CLAUDE.md and README.md (PR 2.5).
    """
    issue_dt = pd.to_datetime(issue_d, format="%b-%Y", errors="coerce")
    last_pymnt_dt = pd.to_datetime(last_pymnt_d, format="%b-%Y", errors="coerce")
    months = (last_pymnt_dt.dt.year - issue_dt.dt.year) * 12 + (last_pymnt_dt.dt.month - issue_dt.dt.month)
    return months


def _parse_emp_length(emp_length: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, 'n/a' -> NaN."""
    def parse_one(val):
        if pd.isna(val):
            return None
        val = str(val).strip().lower()
        if val in ("n/a", "nan", ""):
            return None
        if val.startswith("<"):
            return 0.0
        match = re.search(r"(\d+)", val)
        return float(match.group(1)) if match else None

    return emp_length.apply(parse_one)


def load_raw(path: Path, logger: RunLogger) -> pd.DataFrame:
    if not path.exists():
        raise IngestionError(
            f"Raw data file not found at {path}. Download the Lending Club accepted-loans "
            "CSV (see README.md) and place it there, or set RAW_DATA_PATH in .env."
        )
    df = pd.read_csv(path, low_memory=False)
    logger.log("load_raw", rows_in=len(df), source_path=str(path))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IngestionError(
            f"Raw file is missing required columns: {missing}. This ingestion pipeline is "
            "built against the standard Lending Club accepted-loans schema -- confirm the "
            "downloaded file matches (see README.md)."
        )
    return df


def clean_and_scope(df: pd.DataFrame, logger: RunLogger) -> pd.DataFrame:
    rows_start = len(df)

    # Scope 1: grade C-F only
    df = df[df["grade"].isin(SCOPED_GRADES)].copy()
    logger.log("filter_grade", rows_in=rows_start, rows_out=len(df), grades=sorted(SCOPED_GRADES))

    # Scope 2: matured, known-outcome loans only
    rows_before_status = len(df)
    df = df[df["loan_status"].isin(KEEP_STATUSES)].copy()
    logger.log(
        "filter_loan_status",
        rows_in=rows_before_status,
        rows_out=len(df),
        kept_statuses=sorted(KEEP_STATUSES),
    )

    # Scope 3: platform-maturity filter -- exclude LC's earliest, immature years
    rows_before_maturity = len(df)
    issue_year = _parse_issue_year(df["issue_d"])
    df = df[issue_year >= MIN_ISSUE_YEAR].copy()
    logger.log(
        "filter_platform_maturity",
        rows_in=rows_before_maturity,
        rows_out=len(df),
        min_issue_year=MIN_ISSUE_YEAR,
    )

    # Parse fields
    df["term_months"] = _parse_term(df["term"])
    df["int_rate_frac"] = _parse_int_rate(df["int_rate"])
    df["emp_length_years"] = _parse_emp_length(df["emp_length"])
    df["revol_util_frac"] = _parse_revol_util(df["revol_util"])
    df["months_on_book"] = _compute_months_on_book(df["issue_d"], df["last_pymnt_d"])

    # Derived label: 1 = defaulted/charged off, 0 = fully paid
    df["defaulted"] = df["loan_status"].isin(BAD_STATUSES).astype(int)

    # Data-quality filter: months_on_book must be non-negative and parseable.
    # A negative value means last_pymnt_d predates issue_d, which is either a
    # data error or an edge case (e.g. a payment reversal) -- not usable for
    # a time-to-event calculation either way.
    rows_before_months = len(df)
    df = df[df["months_on_book"].notna() & (df["months_on_book"] >= 0)].copy()
    logger.log(
        "filter_invalid_months_on_book",
        rows_in=rows_before_months,
        rows_out=len(df),
    )

    # Drop rows where key numeric fields failed to parse or are missing
    rows_before_na = len(df)
    key_fields = ["loan_amnt", "term_months", "int_rate_frac", "annual_inc", "dti",
                  "fico_range_low", "fico_range_high", "revol_util_frac",
                  "delinq_2yrs", "open_acc", "pub_rec", "inq_last_6mths",
                  "mort_acc", "total_acc"]
    df = df.dropna(subset=key_fields).copy()
    logger.log("drop_incomplete_rows", rows_in=rows_before_na, rows_out=len(df), fields=key_fields)

    # Dedupe (Lending Club exports occasionally contain duplicate rows)
    rows_before_dedupe = len(df)
    df = df.drop_duplicates()
    logger.log("dedupe", rows_in=rows_before_dedupe, rows_out=len(df))

    # Average FICO, used throughout downstream RAROC/parity logic
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    logger.log("clean_and_scope_complete", rows_final=len(df))
    return df


def run_ingestion(raw_path: Path = RAW_DATA_PATH,
                   processed_path: Path = PROCESSED_DATA_PATH,
                   log_path: Path = LOG_PATH) -> pd.DataFrame:
    logger = RunLogger(log_path)
    try:
        df = load_raw(raw_path, logger)
        df = clean_and_scope(df, logger)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path, index=False)
        logger.log("write_processed", path=str(processed_path), rows=len(df))
        return df
    except IngestionError as e:
        logger.log("ingestion_failed", error=str(e))
        raise
    finally:
        logger.flush()


if __name__ == "__main__":
    result = run_ingestion()
    print(f"Ingestion complete: {len(result):,} loans written to {PROCESSED_DATA_PATH}")
