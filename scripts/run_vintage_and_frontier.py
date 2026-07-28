"""
Runs the vintage loss-emergence curve and the symmetric approval/return
frontier against the real cleaned Lending Club dataset.

Usage:
    python -m scripts.run_vintage_and_frontier
"""
import pandas as pd

from src.features import parse_issue_date, prepare_features, time_based_split
from src.frontier import best_point, compute_frontier
from src.models import train_challenger
from src.vintage import compute_vintage_curve, vintage_summary

SPLIT_DATE = "2015-01-01"


def main():
    print("Loading cleaned dataset...")
    df = pd.read_parquet("data/processed/loans_clean.parquet")
    print(f"  {len(df):,} loans loaded")

    df = parse_issue_date(df)
    df = prepare_features(df)

    print("\n--- Vintage loss-emergence summary (by issue year) ---")
    summary = vintage_summary(df)
    print(summary.to_string(index=False))

    print("\n(Full month-by-month curve available via compute_vintage_curve(df) "
          "-- not printed here, it's long. Use it directly for charting.)")
    curve = compute_vintage_curve(df)  # noqa: F841 -- computed to confirm it runs cleanly

    train, test = time_based_split(df, cutoff_date=SPLIT_DATE)
    print(f"\nTrain: {len(train):,} loans (before {SPLIT_DATE})")
    print(f"Test:  {len(test):,} loans (on/after {SPLIT_DATE})")

    print("\nTraining challenger (logistic regression PD model)...")
    model = train_challenger(train)

    print("\n--- Approval/return frontier (symmetric: both policies swept) ---")
    frontier = compute_frontier(model, train, test)

    champ_best = best_point(frontier, "champion")
    chall_best = best_point(frontier, "challenger")

    print(f"\nChampion's best point on its own frontier:")
    print(f"  FICO cutoff: {champ_best['champion_fico_cutoff']:.0f}")
    print(f"  Approval rate: {champ_best['champion_approval_rate']*100:.1f}%")
    print(f"  RAROC: {champ_best['champion_raroc']*100:.1f}%")

    print(f"\nChallenger's best point on its own frontier:")
    print(f"  PD threshold: {chall_best['challenger_pd_threshold']:.4f}")
    print(f"  Approval rate: {chall_best['challenger_approval_rate']*100:.1f}%")
    print(f"  RAROC: {chall_best['challenger_raroc']*100:.1f}%")

    print(f"\n--- Fair, symmetric conclusion ---")
    print(f"  Champion's best RAROC:    {champ_best['champion_raroc']*100:.1f}% "
          f"at {champ_best['champion_approval_rate']*100:.1f}% approval")
    print(f"  Challenger's best RAROC:  {chall_best['challenger_raroc']*100:.1f}% "
          f"at {chall_best['challenger_approval_rate']*100:.1f}% approval")
    delta = (chall_best['challenger_raroc'] - champ_best['champion_raroc']) * 100
    print(f"  Delta (challenger best - champion best): {delta:+.1f} pts")


if __name__ == "__main__":
    main()
