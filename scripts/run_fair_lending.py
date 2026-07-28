"""
Runs the fair-lending parity screen against the real cleaned Lending Club
dataset, across all three policies: champion, volume-matched challenger,
and RAROC-optimized challenger.

*** READ CLAUDE.md AND src/fair_lending.py'S MODULE DOCSTRING BEFORE USING
THIS OUTPUT FOR ANYTHING. *** This uses a FABRICATED, illustrative-only
state-level probability table -- NOT real Census demographic data. It
demonstrates the MECHANICS of a four-fifths parity screen and a single-leg
(geography-only) BISG-style proxy assignment. It is not evidence of real
disparate impact on this or any real population.

Usage:
    python -m scripts.run_fair_lending
"""
import pandas as pd

from src.fair_lending import assign_geography_proxy, run_parity_screen
from src.features import parse_issue_date, prepare_features, time_based_split
from src.models import (
    DEFAULT_TARGET_APPROVAL_RATE,
    champion_decision_volume_matched,
    challenger_decision_raroc_optimized,
    challenger_decision_volume_matched,
    train_challenger,
)

SPLIT_DATE = "2015-01-01"
TARGET_APPROVAL_RATE = DEFAULT_TARGET_APPROVAL_RATE


def main():
    print("=" * 70)
    print("FAIR-LENDING PARITY SCREEN -- ILLUSTRATIVE ONLY, NOT EVIDENTIARY")
    print("Uses a FABRICATED state-level probability table, not real Census data.")
    print("Demonstrates the four-fifths MECHANISM, not a real disparate-impact finding.")
    print("See CLAUDE.md and src/fair_lending.py for full context.")
    print("=" * 70)

    print("\nLoading cleaned dataset...")
    df = pd.read_parquet("data/processed/loans_clean.parquet")
    print(f"  {len(df):,} loans loaded")

    df = parse_issue_date(df)
    df = prepare_features(df)
    df = assign_geography_proxy(df)
    group_counts = df["protected_class_proxy"].value_counts()
    print(f"  Synthetic proxy assignment: {group_counts.to_dict()}")

    train, test = time_based_split(df, cutoff_date=SPLIT_DATE)
    print(f"  Train: {len(train):,} loans (before {SPLIT_DATE})")
    print(f"  Test:  {len(test):,} loans (on/after {SPLIT_DATE})")

    print("\nTraining challenger (logistic regression PD model)...")
    model = train_challenger(train)

    champ_mask = champion_decision_volume_matched(test, train, TARGET_APPROVAL_RATE)
    vm_mask = challenger_decision_volume_matched(model, test, train, TARGET_APPROVAL_RATE)
    raroc_mask = challenger_decision_raroc_optimized(model, test, train)

    policy_masks = {
        "champion": champ_mask,
        "challenger_volume_matched": vm_mask,
        "challenger_raroc_optimized": raroc_mask,
    }

    print("\n--- Parity screen results (test set) ---")
    results = run_parity_screen(test, policy_masks)
    print(results.to_string(index=False))

    print("\n--- Flags ---")
    for _, row in results.iterrows():
        status = "FLAGGED (below four-fifths threshold)" if row["flagged"] else "passes four-fifths threshold"
        print(f"  {row['policy']}: ratio={row['four_fifths_ratio']:.3f} -- {status}")

    print("\nReminder: every number above comes from a FABRICATED illustrative proxy.")
    print("This shows how the screen works -- it says nothing real about this or any actual population.")


if __name__ == "__main__":
    main()
