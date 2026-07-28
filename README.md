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
- [x] **PR 2 — Champion/challenger model + eval harness**: rule-based cutoff vs. trained PD model, time-based split, RAROC on realized outcomes.
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

### PR 2 description (champion/challenger model + eval harness)

**What**: Adds `src/features.py` (issue-date parsing, time-based train/test split, feature
prep) and `src/models.py` (champion = rule-based FICO cutoff, no model; challenger = trained
logistic-regression PD model; RAROC computation; AUC/Gini/KS eval).

**Why**: Champion vs. challenger here means the real, industry-standard thing — the current
rule-based policy vs. a proposed statistical improvement — not two competing model types
racing each other (that comparison already exists in Credit-Risk-Monitor). RAROC for both
policies is computed from **realized, historical outcomes** on a held-out time-based test
set, not from either policy's own predicted probability — this makes the two directly
comparable on the same real ground truth rather than trusting either one's self-assessed risk.

**Key decisions locked in this PR**:
- Time-based split is mandatory (`issue_d` cutoff date), never random — loan performance and
  applicant mix both drift 2007–2018, a random split would leak future information into training.
- Champion's FICO cutoff (`CHAMPION_FICO_CUTOFF = 660`) is a stated, documented assumption
  representing plausible current practice — not derived from any real institution's actual
  policy. Flagged as an ask-first item in `CLAUDE.md` if this needs to change.
- RAROC loss is computed from the real `defaulted` outcome and real `int_rate_frac`/`loan_amnt`
  — no synthetic or model-estimated numbers feed the historical P&L.

**Tests**: `tests/test_features.py` and `tests/test_models.py`, both against a synthetic
fixture (`tests/conftest.py`) with a real, learnable FICO/default relationship — verifies the
model actually trains, the time split has no overlap, and RAROC/eval metrics behave correctly
at the boundaries (e.g. zero approvals, tighter cutoff never approves more).

**Out of scope for this PR**: vintage curve, approval/return frontier, fair-lending screen,
dashboard — those are PR 3+.
