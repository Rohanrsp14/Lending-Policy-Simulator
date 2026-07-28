"""
Runs champion vs. challenger on the real cleaned Lending Club dataset
(data/processed/loans_clean.parquet, produced by src/data_loader.py) and
prints the RAROC comparison and challenger eval metrics.

IMPORTANT: This dataset contains only ACCEPTED loans (Lending Club does not
publish rejected applications), so the population here has already been
screened by LC's own real underwriting before we ever see it. A fixed FICO
cutoff can therefore be non-binding -- e.g. everyone might already clear 660.
This is the "reject inference" limitation, and it's a known, documented gap
in accepted-loans-only data (see CLAUDE.md and README.md), not a bug.

To get a meaningful comparison despite this, champion and challenger are
both calibrated to the SAME approval rate (a swap-set analysis: same volume,
different mix of who gets approved) rather than an arbitrary absolute cutoff.

Usage:
    python -m scripts.run_evaluation
"""
import pandas as pd

from src.features import parse_issue_date, prepare_features, time_based_split
from src.models import (
    DEFAULT_TARGET_APPROVAL_RATE,
    calibrate_fico_cutoff,
    calibrate_pd_threshold,
    champion_decision_volume_matched,
    challenger_decision_volume_matched,
    compute_raroc,
    evaluate_challenger,
    train_challenger,
)

SPLIT_DATE = "2015-01-01"
TARGET_APPROVAL_RATE = DEFAULT_TARGET_APPROVAL_RATE  # 0.85 -- decline riskiest ~15%


def main():
    print("Loading cleaned dataset...")
    df = pd.read_parquet("data/processed/loans_clean.parquet")
    print(f"  {len(df):,} loans loaded")

    df = parse_issue_date(df)
    df = prepare_features(df)

    train, test = time_based_split(df, cutoff_date=SPLIT_DATE)
    print(f"  Train: {len(train):,} loans (before {SPLIT_DATE})")
    print(f"  Test:  {len(test):,} loans (on/after {SPLIT_DATE})")

    print("\nTraining challenger (logistic regression PD model)...")
    model = train_challenger(train)

    print("\n--- Challenger ranking quality (held-out test set) ---")
    eval_metrics = evaluate_challenger(model, test)
    for k, v in eval_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    fico_cutoff = calibrate_fico_cutoff(train, TARGET_APPROVAL_RATE)
    pd_threshold = calibrate_pd_threshold(model, train, TARGET_APPROVAL_RATE)
    print(f"\nCalibrated on training population to target {TARGET_APPROVAL_RATE*100:.0f}% approval:")
    print(f"  Champion FICO cutoff: {fico_cutoff:.0f}")
    print(f"  Challenger PD threshold: {pd_threshold:.4f}")

    champ_mask = champion_decision_volume_matched(test, train, TARGET_APPROVAL_RATE)
    chall_mask = challenger_decision_volume_matched(model, test, train, TARGET_APPROVAL_RATE)

    champ_metrics = compute_raroc(test, champ_mask)
    chall_metrics = compute_raroc(test, chall_mask)

    print(f"\n--- RAROC comparison on held-out test set (volume-matched) ---")
    print(f"\nChampion (FICO cutoff >= {fico_cutoff:.0f}):")
    print(f"  Approved: {champ_metrics['n']:,} ({champ_metrics['approval_rate']*100:.1f}%)")
    print(f"  Loss rate: {champ_metrics['loss_rate']*100:.2f}%")
    print(f"  RAROC: {champ_metrics['raroc']*100:.1f}%")

    print(f"\nChallenger (PD threshold <= {pd_threshold:.4f}):")
    print(f"  Approved: {chall_metrics['n']:,} ({chall_metrics['approval_rate']*100:.1f}%)")
    print(f"  Loss rate: {chall_metrics['loss_rate']*100:.2f}%")
    print(f"  RAROC: {chall_metrics['raroc']*100:.1f}%")

    print(f"\n--- Delta (Challenger - Champion), at matched approval volume ---")
    print(f"  Volume: {chall_metrics['n'] - champ_metrics['n']:+,} loans")
    print(f"  Approval rate: {(chall_metrics['approval_rate'] - champ_metrics['approval_rate'])*100:+.1f} pts")
    print(f"  Loss rate: {(chall_metrics['loss_rate'] - champ_metrics['loss_rate'])*100:+.2f} pts")
    print(f"  RAROC: {(chall_metrics['raroc'] - champ_metrics['raroc'])*100:+.1f} pts")


if __name__ == "__main__":
    main()
