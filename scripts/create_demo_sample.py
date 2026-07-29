"""
Creates a random ~100K-loan sample of the full cleaned dataset, for the
public Streamlit Community Cloud deployment. The full dataset (685,806
loans) is too large to commit and isn't redistributed here -- Lending
Club's terms don't clearly cover republishing a near-complete copy of
their data at that scale. A smaller, clearly-labeled sample is the safer,
standard practice for a public ML portfolio demo.

Every finding in docs/MODEL_VALIDATION.md is based on the FULL dataset,
run locally -- this sample is for the public demo's interactivity only.

Usage:
    python -m scripts.create_demo_sample
"""
import pandas as pd

SAMPLE_SIZE = 100_000
SEED = 20260101

INPUT_PATH = "data/processed/loans_clean.parquet"
OUTPUT_PATH = "data/processed/loans_demo_sample.parquet"


def main():
    print(f"Loading full dataset from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  {len(df):,} loans loaded")

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    print(f"  Sampled {len(sample):,} loans (random, seed={SEED})")

    # Sanity check: confirm the sample still spans the full vintage range
    # meaningfully (a genuinely broken sample would show a near-empty year).
    year_counts = pd.to_datetime(sample["issue_d"], format="%b-%Y").dt.year.value_counts().sort_index()
    print("\nSampled loans per issue year (sanity check):")
    print(year_counts.to_string())

    sample.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()