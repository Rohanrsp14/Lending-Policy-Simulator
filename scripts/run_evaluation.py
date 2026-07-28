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

Shows THREE comparisons:
  1. Champion -- rule-based FICO cutoff, volume-matched to a target approval rate
  2. Challenger (volume-matched) -- PD model, calibrated to the SAME approval
     rate as champion (a swap-set analysis: same volume, different mix)
  3. Challenger (RAROC-optimized) -- PD model, threshold chosen on the training
     set specifically to maximize RAROC, not just match a volume target. This
     surfaces a real finding: a model that reduces default risk doesn't always
     improve RAROC if it disproportionately declines high-margin (high-rate)
     loans along with the risky ones. See CLAUDE.md.

Usage:
    python -m scripts.run_evaluation
"""
import pandas as pd

from src.features import parse_issue_date, prepare_features, time_based_split
from src.models import (
    DEFAULT_TARGET_APPROVAL_RATE,
    calibrate_fico_cutoff,
    calibrate_pd_threshold,
    calibrate_pd_threshold_for_raroc,
    champion_decision_volume_matched,
    challenger_decision_raroc_optimized,
    challenger_decision_volume_matched,
    compute_raroc,
    evaluate_challenger,
    train_challenger,
)

SPLIT_DATE = "2015-01-01"
TARGET_APPROVAL_RATE = DEFAULT_TARGET_APPROVAL_RATE  # 0.85 -- decline riskiest ~15%


def show(name: str, m: dict):
    print(f"\n{name}:")
    print(f"  Approved: {m['n']:,} ({m['approval_rate']*100:.1f}%)")
    print(f"  Loss rate (lifetime): {m['loss_rate']*100:.2f}%")
    print(f"  Annualized revenue: {m['revenue']:,.0f}")
    print(f"  RAROC: {m['raroc']*100:.1f}%")


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
    vm_threshold = calibrate_pd_threshold(model, train, TARGET_APPROVAL_RATE)
    raroc_threshold = calibrate_pd_threshold_for_raroc(model, train)

    print(f"\nCalibrated on training population:")
    print(f"  Champion FICO cutoff (target {TARGET_APPROVAL_RATE*100:.0f}% approval): {fico_cutoff:.0f}")
    print(f"  Challenger PD threshold (volume-matched): {vm_threshold:.4f}")
    print(f"  Challenger PD threshold (RAROC-optimized): {raroc_threshold:.4f}")

    champ_mask = champion_decision_volume_matched(test, train, TARGET_APPROVAL_RATE)
    vm_mask = challenger_decision_volume_matched(model, test, train, TARGET_APPROVAL_RATE)
    raroc_opt_mask = challenger_decision_raroc_optimized(model, test, train)

    champ_metrics = compute_raroc(test, champ_mask)
    vm_metrics = compute_raroc(test, vm_mask)
    raroc_opt_metrics = compute_raroc(test, raroc_opt_mask)

    print(f"\n--- Three-way RAROC comparison on held-out test set ---")
    show(f"Champion (FICO cutoff >= {fico_cutoff:.0f})", champ_metrics)
    show(f"Challenger, volume-matched (PD <= {vm_threshold:.4f})", vm_metrics)
    show(f"Challenger, RAROC-optimized (PD <= {raroc_threshold:.4f})", raroc_opt_metrics)

    print(f"\n--- Delta: RAROC-optimized challenger vs. Champion ---")
    print(f"  Volume: {raroc_opt_metrics['n'] - champ_metrics['n']:+,} loans")
    print(f"  Approval rate: {(raroc_opt_metrics['approval_rate'] - champ_metrics['approval_rate'])*100:+.1f} pts")
    print(f"  Loss rate: {(raroc_opt_metrics['loss_rate'] - champ_metrics['loss_rate'])*100:+.2f} pts")
    print(f"  RAROC: {(raroc_opt_metrics['raroc'] - champ_metrics['raroc'])*100:+.1f} pts")


if __name__ == "__main__":
    main()
