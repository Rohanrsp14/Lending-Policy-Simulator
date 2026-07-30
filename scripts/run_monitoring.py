"""
Runs PSI (Population Stability Index) drift monitoring, comparing the
training population (pre-2015 originations) against the test population
(2015+ originations) -- a real before/after check using this project's
own time-based split, not a synthetic demonstration.

Usage:
    python -m scripts.run_monitoring
"""
import pandas as pd

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    parse_issue_date,
    prepare_features,
    time_based_split,
)
from src.monitor import compute_psi_report

SPLIT_DATE = "2015-01-01"


def main():
    print("Loading cleaned dataset...")
    df = pd.read_parquet("data/processed/loans_clean.parquet")
    print(f"  {len(df):,} loans loaded")

    df = parse_issue_date(df)
    df = prepare_features(df)
    train, test = time_based_split(df, cutoff_date=SPLIT_DATE)
    print(f"  Train (baseline): {len(train):,} loans (before {SPLIT_DATE})")
    print(f"  Test (checked for drift): {len(test):,} loans (on/after {SPLIT_DATE})")

    print("\nComputing PSI for every feature (train = baseline, test = checked population)...")
    report = compute_psi_report(train, test, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    print("\n--- PSI report (sorted by most-drifted first) ---")
    print(report.to_string(index=False))

    print("\n--- Summary ---")
    for status in ["significant_shift", "moderate_shift", "stable"]:
        features = report[report["status"] == status]["feature"].tolist()
        if features:
            print(f"  {status} ({len(features)}): {', '.join(features)}")

    significant = report[report["status"] == "significant_shift"]
    if len(significant) > 0:
        print(f"\n{len(significant)} feature(s) show significant population drift between "
              f"train and test periods -- this is a real, additional explanation for the "
              f"model's modest AUC (docs/MODEL_VALIDATION.md Section 5): the population it's "
              f"scoring has genuinely shifted from what it was trained on.")
    else:
        print("\nNo features show significant drift -- the train and test populations are "
              "reasonably stable relative to each other.")


if __name__ == "__main__":
    main()
