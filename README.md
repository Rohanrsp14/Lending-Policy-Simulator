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
- [x] **PR 2.1 — Volume-matched calibration + expanded risk features**: fixes the reject-inference finding from running PR 2 against real data.
- [x] **PR 2.2 — Amortized revenue, more features, platform-maturity scope**: fixes 3 limitations found reviewing PR 2.1's real results.
- [x] **PR 2.3 — RAROC annualization fix**: fixes a real bug (RAROC >100%) surfaced by running PR 2.2 against real data.
- [x] **PR 2.4 — RAROC-optimized threshold selection**: fixes a real finding — volume-matching isn't the same as RAROC-optimal.
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

### PR 2.1 description (volume-matched calibration + expanded risk features)

**What**: Running PR 2 against the real 708,134-loan dataset surfaced a real finding: a fixed
FICO cutoff of 660 approved 100% of the held-out test set, because this dataset contains only
Lending Club's already-accepted loans — the population was already screened by LC's own real
underwriting before we ever see it (a known credit-modeling limitation called **reject
inference**, documented in `CLAUDE.md`). This PR fixes the comparison to be meaningful despite
that: champion and challenger are now both calibrated to the same target approval rate (85%
by default) on the training population — a swap-set analysis (same volume, different mix)
rather than an arbitrary absolute threshold. Also adds five real risk features already present
in the raw data (`revol_util`, `delinq_2yrs`, `open_acc`, `pub_rec`, `inq_last_6mths`) to
strengthen the challenger model's ranking quality.

**Why**: An arbitrary fixed cutoff either over- or under-states the comparison depending on
where a given population's score distribution happens to sit — calibrating both policies to
the same approval volume is the standard real-world "swap set" framing and makes the RAROC
delta a genuine reflection of selection quality, not an artifact of an arbitrary number.

**Tests**: 4 new tests verifying the calibration functions actually hit their target approval
rate (within tolerance) and that champion/challenger land on comparable approval volumes when
both are volume-matched.

**Out of scope for this PR**: still PR 3+ for vintage curve, frontier, fair-lending, dashboard.

### PR 2.2 description (amortized revenue, more features, platform-maturity scope)

**What**: Fixes three real limitations identified reviewing PR 2.1's actual output on the
708K-loan dataset:
1. `src/models.py::amortized_interest` replaces a flat `loan_amnt * rate * fudge_factor`
   revenue approximation with a real fixed-payment installment amortization formula.
   Documented limitation: still assumes full-term repayment, which overstates revenue on
   loans that default early — intentionally deferred to PR 3's vintage/time-on-book work,
   not silently ignored.
2. Adds `home_ownership`, `verification_status`, `mort_acc`, `total_acc` — real fields
   already in the raw data — to strengthen the challenger model (AUC was a modest 0.63).
3. Scopes data to loans issued 2012 or later (`MIN_ISSUE_YEAR` in `src/data_loader.py`) —
   Lending Club's 2007-2011 originations were a small, immature platform with materially
   different underwriting than the post-2012 period; mixing regimes added noise.

**Why**: An interviewer reviewing the RAROC math would catch the amortization gap
immediately, and the AUC/regime-mixing issues were visible the moment PR 2.1 ran against
real data — better to name and fix these now than present numbers that don't hold up to a
second look.

**Tests**: 4 new tests (32 total) — amortized interest validated against a manually
computed example, confirms longer terms produce more total interest, confirms RAROC now
uses the amortized figure, and confirms the platform-maturity filter excludes pre-2012 loans.

**Out of scope for this PR**: fully correcting revenue for early-default truncation (needs
per-loan time-to-default, which is PR 3's job) and any model-type change (still logistic
regression, by design — see PR 2's rationale for why this stays a rules-vs-model comparison
rather than a model-vs-model one).

### PR 2.3 description (RAROC annualization fix — a real bug)

**What**: Running PR 2.2 against the real dataset produced RAROC of 201.5% (champion) and
154.3% (challenger) — not believable numbers, and a real bug rather than a documented
limitation. `amortized_interest()` (added in PR 2.2) returns full-loan-life revenue totals
(3-5 years' worth), but `capital` and `opex` are one-time charges meant to represent an
*annual* basis — the standard RAROC convention. Comparing a multi-year total against a
one-year capital base is a unit mismatch, and it's what produced RAROC over 100%.

**Fix**: `compute_raroc` now divides both revenue and realized loss by each loan's own
term-in-years before summing, putting every quantity feeding RAROC on the same annual basis
as the capital charge. `loss_rate` in the output remains a simple lifetime figure (unannualized)
since that's the more intuitive way to read "what fraction of approved balance was lost."

**Why this is worth naming explicitly rather than quietly patching**: this is exactly the kind
of math error a real risk team's model-validation review would catch before signing off on a
RAROC number — catching and documenting it here is a stronger interview story than either
hiding it or never having made it in the first place.

**Tests**: updated the two PR 2.2 revenue/loss tests to expect annualized figures, and added
`test_compute_raroc_is_annualized_not_lifetime_total` as an explicit regression guard — asserts
RAROC stays under 100% on the synthetic fixture, so this specific bug class can't silently
reappear.

**Out of scope for this PR**: still PR 3+ for vintage curve, frontier, fair-lending, dashboard.

### PR 2.4 description (RAROC-optimized threshold selection)

**What**: Running PR 2.3 against real data surfaced a real, explainable finding: the
volume-matched challenger had a lower loss rate than champion (15.30% vs. 17.31%) but a
*worse* RAROC (-7.3% vs. 0.1%). Diagnosis (checking approved-population averages) confirmed
why: the challenger's approved set had a lower average interest rate, smaller average loan
amount, and shorter average term than champion's — a PD model that reduces default risk will
naturally tend to decline higher-rate (higher-revenue) loans too, since Lending Club's own
rate assignment already reflects their risk view. Optimizing a ranking metric (PD) isn't the
same as optimizing the real economic objective (RAROC).

**Fix/addition**: `sweep_pd_thresholds` evaluates RAROC across a range of PD thresholds on
the training set (no leakage), and `calibrate_pd_threshold_for_raroc` selects the threshold
that directly maximizes RAROC rather than matching a volume target.
`challenger_decision_raroc_optimized` applies it. `scripts/run_evaluation.py` now prints all
three policies side by side: champion, volume-matched challenger, and RAROC-optimized
challenger — so the improvement (or lack of one) from actually optimizing for the right
objective is visible directly.

**Why this matters for the interview story**: this is a genuinely sophisticated point —
"a model that improves default prediction doesn't automatically improve risk-adjusted return
if it disproportionately declines your highest-margin loans along with the risky ones; you
have to calibrate against the actual economic objective, not just a ranking metric." Finding
and fixing this, rather than presenting the volume-matched number as the final answer, is a
stronger result than either outcome alone.

**Tests**: 3 new tests (36 total) — verifies the sweep returns the expected structure,
confirms `calibrate_pd_threshold_for_raroc` actually selects the sweep's best row, and
confirms the RAROC-optimized threshold never underperforms volume-matching on the population
it was optimized on (a built-by-construction guarantee, checked as a regression test).

**Out of scope for this PR**: still PR 3+ for vintage curve, frontier, fair-lending, dashboard.
