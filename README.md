# Lending Policy Simulator

A champion/challenger consumer-lending policy simulator built on real, public **Lending Club**
loan-level data, scoped to a near-prime/subprime population representative of a Regional
Finance-tier installment lender. Computes a full P&L waterfall to risk-adjusted return
(RAROC), vintage loss emergence, an approval/return frontier, and a fair-lending parity
screen — deployed as a Streamlit dashboard.

**This is a portfolio/analytical project, not a production underwriting system.** See
[CLAUDE.md](CLAUDE.md) for full data provenance and project rules.

## Status

Built incrementally via small, sprint-sized PRs rather than one large drop:

- [x] **PR 1 — Data ingestion**: load, filter, clean, and label the raw Lending Club export.
- [ ] PR 2 — Champion/challenger RAROC model + eval harness
- [ ] PR 3 — Vintage loss curve + approval/return frontier
- [ ] PR 4 — Fair-lending parity screen
- [ ] PR 5 — Streamlit dashboard
- [ ] PR 6 — CI polish, docs, deploy

## Dataset

| Dataset | Description | Source |
|---|---|---|
| Lending Club Accepted Loans | Loan-level consumer installment lending data, 2007–2020 | Commonly distributed via Kaggle: `wordsforthewise/lending-club` (search "Lending Club Loan Data" if that slug changes) |

Not included in this repo (large file, license terms require direct download). Download the
accepted-loans CSV and place it at `data/raw/accepted_2007_to_2018Q4.csv` (or update the path
in `.env`).

**v1 scope**: filtered to loan grades **C–F** (near-prime/subprime, proxy for the Regional
Finance lending tier) and to loans with a matured, known outcome (`Fully Paid` or
`Charged Off`/`Default` only — see [CLAUDE.md](CLAUDE.md) for why `Current`/`Issued`/`Late`
loans are excluded).

## Folder structure

```
lending-policy-simulator/
├── data/
│   ├── raw/            # Original Lending Club CSV (gitignored, download separately)
│   └── processed/      # Cleaned/filtered dataset (gitignored, regenerable)
├── src/
│   ├── data_loader.py  # Ingestion: load, filter, clean, label
│   ├── features.py     # (PR 2) RAROC/loss/capital feature engineering
│   ├── models.py        # (PR 2) Champion/challenger cutoff logic
│   ├── vintage.py       # (PR 3) Vintage loss emergence
│   ├── frontier.py      # (PR 3) Approval/return frontier sweep
│   └── fair_lending.py  # (PR 4) Four-fifths parity screen
├── dashboard/
│   └── app.py            # (PR 5) Streamlit dashboard
├── logs/                  # Structured ingestion/run logs (gitignored)
├── tests/
│   └── test_data_loader.py
├── .github/workflows/
│   └── ci.yml             # Lint + test on every push/PR
├── .env.example
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd lending-policy-simulator

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

cp .env.example .env         # then set RAW_DATA_PATH if different from the default
```

Place the downloaded Lending Club CSV in `data/raw/`, then run the ingestion pipeline:

```bash
python -m src.data_loader
```

This produces `data/processed/loans_clean.parquet` and writes a structured run log to
`logs/ingestion_runs.jsonl`.

Run tests any time with:

```bash
pytest
```

## Contribution workflow (for this build)

Each feature ships as its own small PR against `main`, sized like a real sprint ticket:

1. Branch off `main`: `git checkout -b feature/<short-name>`
2. Build the increment + its tests.
3. `pytest` passes locally, CI passes on push.
4. Open a PR with a short description of what changed and why — see PR 1's description
   below as the template for future ones.
5. Merge once reviewed.

### PR 1 description (data ingestion)

**What**: Adds `src/data_loader.py` — loads the raw Lending Club CSV, filters to grades C–F
and matured outcomes only, parses `term`/`int_rate`/`emp_length` into numeric fields, derives
a binary default label, dedupes, and writes a cleaned Parquet file plus a structured JSONL
run log (row counts at every filter step).

**Why**: This is the foundation every later PR (RAROC model, vintage curve, frontier, fair-lending
screen, dashboard) reads from — locking the scope and cleaning logic first, with tests, avoids
every downstream PR quietly re-deriving its own inconsistent version of "clean data."

**Tests**: `tests/test_data_loader.py` uses a small in-memory synthetic CSV fixture (not the
real 12GB+ file) so the cleaning/parsing/filtering logic is verified deterministically and
fast, independent of whether the real data file is present.

**Out of scope for this PR**: model logic, RAROC calculation, dashboard — those are PR 2+.
