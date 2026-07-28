"""
Runs the RAROC sensitivity analysis against the real cleaned Lending Club
dataset -- stress-tests whether champion's RAROC advantage over challenger
(docs/MODEL_VALIDATION.md Section 6) holds across a plausible range of the
LGD/opex/capital assumptions, one at a time.

Usage:
    python -m scripts.run_sensitivity
"""
import numpy as np
import pandas as pd

from src.features import parse_issue_date, prepare_features, time_based_split
from src.models import train_challenger
from src.sensitivity import run_sensitivity, summarize_robustness

SPLIT_DATE = "2015-01-01"

# Finer grid than compute_frontier's own default (25 points) -- validates
# that the found optimum approval rate is a genuine peak, not an artifact
# of a coarse grid. See CLAUDE.md.
FINE_QUANTILES = np.linspace(0.50, 0.99, 100)


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

    print("\nRunning sensitivity sweep across LGD, opex rate, and capital rate...")
    print(f"(Using a fine {len(FINE_QUANTILES)}-point grid per sweep, to confirm the optimum "
          f"approval rate is a genuine peak, not a coarse-grid artifact -- this will take a few minutes.)")
    sensitivity = run_sensitivity(model, train, test, quantiles=FINE_QUANTILES)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    for param in ["lgd", "opex_rate", "capital_rate"]:
        print(f"\n--- Sweeping {param} (others held at base case) ---")
        subset = sensitivity[sensitivity["swept_parameter"] == param][
            ["parameter_value", "champion_best_raroc", "champion_best_approval_rate",
             "challenger_best_raroc", "challenger_best_approval_rate",
             "delta_challenger_minus_champion", "champion_wins"]
        ]
        print(subset.to_string(index=False))

    print("\n--- Robustness summary ---")
    summary = summarize_robustness(sensitivity)
    print(summary.to_string(index=False))

    print("\n--- Plain-language read ---")
    for _, row in summary.iterrows():
        if row["robustness"] == "champion_always_wins":
            print(f"  {row['swept_parameter']}: champion wins across the ENTIRE tested range -- a robust finding.")
        elif row["robustness"] == "challenger_always_wins":
            print(f"  {row['swept_parameter']}: challenger wins across the ENTIRE tested range -- also robust, opposite direction.")
        else:
            print(f"  {row['swept_parameter']}: the conclusion FLIPS somewhere in the tested range -- "
                  f"this is the number Treasury/Finance would need to nail down before trusting either conclusion.")

    print("\n--- Optimum stability check (fine grid) ---")
    champ_rates = sensitivity["champion_best_approval_rate"].round(4).unique()
    chall_rates = sensitivity["challenger_best_approval_rate"].round(4).unique()
    print(f"  Champion's best approval rate, across ALL sweeps: {sorted(champ_rates)}")
    print(f"  Challenger's best approval rate, across ALL sweeps: {sorted(chall_rates)}")
    if len(champ_rates) == 1 and len(chall_rates) == 1:
        print("  Both optima are stable single points even at a fine grid -- confirms genuine, "
              "sharp peaks (proportional cost scaling doesn't move where the optimum sits), "
              "not an artifact of the earlier coarser 25-point grid.")
    else:
        print("  The optimum MOVED under the finer grid or across parameter values -- "
              "worth a closer look before treating the earlier single-point result as settled.")


if __name__ == "__main__":
    main()
