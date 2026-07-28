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
