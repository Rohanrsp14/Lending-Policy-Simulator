# CLAUDE.md — Lending Policy Simulator

Project context and build rules for anyone (human or Claude Code) working on this repo.

## What this is

A champion/challenger consumer-lending policy simulator built on real, public Lending Club
loan-level data, scoped to a near-prime/subprime population representative of a Regional
Finance-tier installment lender. Computes a P&L waterfall to RAROC, vintage loss emergence,
an approval/return frontier, and a fair-lending parity screen — deployed as a Streamlit app.

**This is a portfolio/analytical project, not a production underwriting system.** Nothing
here should be presented as, or used as, an actual credit decisioning tool.

## Data provenance

- **Source**: Lending Club accepted loans, public dataset (commonly distributed via Kaggle,
  e.g. `wordsforthewise/lending-club`). Not included in this repo — download separately and
  place in `data/raw/`.
- **Scope for v1**: filtered to loan grades C–F (near-prime/subprime) as a proxy for the
  Regional Finance lending tier, and to loans with a **matured, known outcome**
  (`Fully Paid` or `Charged Off` / `Default` only — `Current`, `Issued`, and `Late` statuses
  are excluded because their eventual outcome is not yet known and including them would leak
  survivorship bias into the loss-rate calculation).
- Every derived dataset must be traceable back to this raw source. No synthetic rows are
  ever mixed into the real dataset without an explicit, visible flag.

## PR 2.5: real time-to-default data for the vintage curve

PR 1's own scoping decision (matured, known-outcome loans only) meant every loan in this
dataset already has a terminal outcome -- but that also meant we had no field capturing
*when* that outcome occurred, which is required for a genuine vintage loss-emergence curve
(PR 3). Fixed by adding `last_pymnt_d` (last payment date, a standard Lending Club field) and
computing `months_on_book = last_pymnt_d - issue_d` in `src/data_loader.py`. For Charged
Off/Default loans this approximates time-to-default; for Fully Paid loans it's time-to-payoff
(which can be less than `term_months` if prepaid early). Rows where this comes out negative
(a data-quality edge case, `last_pymnt_d` predating `issue_d`) are filtered and logged. This
lets PR 3 build the vintage curve from real per-loan timing rather than reverting to an
illustrative/synthetic maturation curve.

## Known finding addressed in PR 2.4: volume-matching isn't the same as RAROC-optimal

Running PR 2.3 against real data surfaced a real, explainable finding: the volume-matched
challenger had a LOWER loss rate than champion but a WORSE RAROC. Root cause (confirmed by
checking approved-population averages): the PD model's approved set had a lower average
interest rate, smaller average loan amount, and shorter average term than champion's --
because Lending Club's own rate assignment already reflects their view of risk, a model
that reduces default risk will naturally also tend to decline higher-rate (higher-revenue)
loans. This is not a bug -- it's a real illustration that optimizing a ranking metric (PD)
is not the same as optimizing the actual economic objective (RAROC).

**Fix/addition**: `src/models.py::sweep_pd_thresholds` and `calibrate_pd_threshold_for_raroc`
sweep candidate PD thresholds on the TRAINING set only (no leakage) and select the one that
directly maximizes RAROC, rather than matching a target approval volume. `scripts/run_evaluation.py`
now shows all three: champion, volume-matched challenger, and RAROC-optimized challenger.

## Known limitations addressed in PR 2.3

- **RAROC annualization bug, found and fixed**: PR 2.2's amortization fix computed
  revenue and loss as full-loan-life totals (e.g. 3-5 years' worth), but `capital` and
  `opex` remained one-time charges meant to represent an ANNUAL basis. Comparing a
  multi-year total against a one-year capital base produced RAROC over 100% when run
  against real data -- a real bug, not a portfolio quirk. Fixed in `src/models.py::compute_raroc`
  by dividing both revenue and loss by each loan's own term-in-years before summing, so
  every quantity feeding RAROC is on a consistent annual basis. `test_compute_raroc_is_annualized_not_lifetime_total`
  is a regression test guarding against this specific class of bug recurring.

## Known limitations addressed in PR 2.2

- **Revenue is amortized, not flat-multiplied** (`src/models.py::amortized_interest`) --
  a real fixed-payment installment amortization formula, replacing an earlier flat
  `loan_amnt * rate * fudge_factor` approximation. This still assumes every approved
  loan runs to full term, which overstates revenue specifically on loans that default
  early (they stop paying before maturity). Fully correcting this needs each loan's
  actual time-to-default -- deferred intentionally to PR 3's vintage/time-on-book work,
  not silently ignored.
- **Platform-maturity scope**: data is now filtered to loans issued in `MIN_ISSUE_YEAR`
  (2012) or later. Lending Club's 2007-2011 originations were a small, immature platform
  with materially different underwriting and volume than the post-2012 period --
  mixing these regimes into one model added noise. This is a stated, locked scope
  decision (ask-first if it needs to change), not a silent data drop.
- **Expanded features**: `home_ownership`, `verification_status`, `mort_acc`,
  `total_acc` added -- all real fields already present in the raw Lending Club export,
  added to strengthen the challenger model's ranking quality (AUC was modest at 0.63
  with the original feature set).

## Known limitation: reject inference

This dataset contains **only accepted loans** -- Lending Club does not publish
rejected applications. Every loan here was already screened by LC's own real
underwriting before this data was ever published. This means:

- A fixed absolute FICO cutoff can be **non-binding** on this population --
  observed on the real 2007-2018Q4 data: a champion cutoff of 660 approved
  100% of the held-out test set, because the scoped population's FICO floor
  already sits at or above that threshold as an artifact of LC's own real
  acceptance decision, not because 660 is a meaningful decision boundary here.
- This is a well-known credit-modeling limitation called **reject inference**:
  outcome data only exists for approved applicants, so any policy "cutoff"
  defined on that population alone cannot be validated against the applicants
  who were never observed.
- **Mitigation used in this project**: champion and challenger are both
  calibrated to the same target approval rate on the training population
  (`calibrate_fico_cutoff` / `calibrate_pd_threshold` in `src/models.py`) --
  a swap-set analysis (same volume, different mix) rather than an arbitrary
  absolute threshold. This produces a meaningful comparison despite the
  underlying reject-inference gap, but does NOT solve reject inference
  itself -- it is still true that this project cannot say anything about
  applicants LC actually declined.

## Always-do

- Cite the data source and vintage/date range in every output and chart.
- Log every ingestion run (row counts in/out at each filter step, timestamp, source file
  hash if feasible) to `logs/` — structured, not just print statements.
- Write a test for every cleaning/parsing function before wiring it into the pipeline.
- State every simplifying assumption in RAROC/capital/loss formulas explicitly in the README
  and in-app — these are illustrative, not calibrated to Lending Club's or any real
  institution's actual economics.
- Keep the fair-lending parity check's protected-class proxy field clearly labeled as
  **synthetic/illustrative** if Lending Club data itself doesn't include one (it doesn't —
  no demographic fields are in this dataset), so the parity check is never mistaken for a
  real disparate-impact analysis on real applicant demographics.

## Ask-first

- Changing the grade/tier scope (currently C–F).
- Changing the loan-status inclusion/exclusion rule (currently Fully Paid + Charged Off/Default only).
- Adding any new external data source.
- Changing the RAROC/capital formula assumptions.

## Never-do

- Never fabricate or synthesize loan records mixed into the real dataset without a visible flag.
- Never hardcode API keys/secrets — Kaggle credentials go in `.env`, never committed.
- Never claim this tool certifies fair-lending compliance, SR 11-7 model-risk compliance, or
  any real regulatory standard — it borrows the concepts for realistic structure only.
- Never overclaim model/metric accuracy beyond what the eval actually shows.
