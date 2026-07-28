# Model Validation Report — Lending Policy Simulator (Champion/Challenger)

**Model reviewed**: Champion (rule-based FICO cutoff) vs. Challenger (logistic regression PD model)
**Reviewer stance**: Written as an independent, critical second review — not the model builder's own assessment — consistent with SR 11-7's "effective challenge" principle. Where the model looks good, this report says so. Where it doesn't, this report says that too, plainly.
**Data**: 685,806 matured, near-prime/subprime (grade C–F) Lending Club loans, issued 2012–2018Q4
**Status**: **Not approved for production use.** See Section 7 for conditions.

---

## 1. Scope & Purpose

This model was built as a portfolio/methodology exercise, not a production underwriting tool (stated explicitly in `CLAUDE.md` since PR 1). This report evaluates it against the same standard a real model-risk function would apply, specifically so its gaps are documented rather than discovered later by someone else. That distinction — portfolio artifact vs. production-ready — is the single most important fact in this report and should not get lost in any summary of it.

## 2. Model Overview

- **Champion**: a deterministic FICO-score cutoff. No model, no training. Represents a plausible current rule-based underwriting practice.
- **Challenger**: a logistic regression predicting probability of default, trained on: FICO, DTI, annual income, employment length, loan term, loan amount, revolving utilization, delinquency count, open accounts, public records, recent inquiries, mortgage accounts, total accounts, home ownership, verification status, and purpose. **Interest rate was deliberately excluded** after being identified as a confounding feature (Section 7.4).
- Both policies are evaluated identically: RAROC computed from **realized historical outcomes** on a held-out, time-based test set — neither policy grades its own homework.

## 3. Conceptual Soundness Review

**Sound**:
- Time-based train/test split (not random) — correctly avoids leaking future loan vintages into training.
- RAROC computed from realized outcomes, not either policy's self-assessed risk — a real, defensible evaluation design.
- Volume-matched *and* RAROC-optimized comparisons were both built, rather than relying on a single arbitrarily-chosen operating point (Section 6).

**Not sound without qualification**:
- The economic assumptions (55% LGD, 5% opex, 8% capital charge — all flat, all illustrative) are not calibrated to any real institution's actual cost structure. Every RAROC number in this project is only as meaningful as these three constants, and they are currently placeholders, clearly labeled as such in `CLAUDE.md`.
- The model was trained and evaluated entirely on Lending Club's book, not Regional Finance's own originations. Findings here characterize *this dataset*, not any specific company's portfolio.

## 4. Data Review

- **Reject inference**: this dataset contains only accepted loans. Lending Club's own underwriting already screened the population before this project ever saw it. Discovered directly: a fixed FICO cutoff of 660 approved 100% of a held-out test set — not because 660 is meaningless, but because everyone in this population already clears it. **Mitigation**: champion and challenger are calibrated to the same target approval rate (a swap-set design) rather than relying on absolute thresholds, which are not trustworthy on accepted-only data. This does not solve reject inference — it works around its worst symptom.
- **Platform-maturity scope**: loans issued before 2012 were excluded — Lending Club's earliest years were a small, immature platform with different underwriting than the 2012+ period. A stated, deliberate scope decision, not a silent drop.
- **Data-quality filtering**: rows with missing/unparseable key fields, and rows where `last_pymnt_d` predates `issue_d` (a data error, not a modeling case) were dropped and logged at every step (see `logs/ingestion_runs.jsonl`).
- **No bureau-depth features**: this model has no tradeline-level history, no full inquiry history, no collections detail beyond what Lending Club itself publishes. A real production PD model would have materially more feature depth.

## 5. Discriminatory Power

Held-out test set (post-2015 originations), n = 467,419:

| Metric | Value | Assessment |
|---|---|---|
| AUC | 0.6396 | Modest. Meaningfully better than chance, well below the 0.70–0.75+ typical of production subprime PD models with full bureau data. |
| Gini | 0.2791 | Consistent with the AUC — real but limited separation. |
| KS | 0.2019 | Below the 0.30+ often used as a soft internal benchmark for a strong scorecard. |

**Verdict**: the model discriminates risk better than random, but not strongly. This number alone should not be presented as evidence the model is production-ready — see Section 6 for why predictive power and economic value are two different questions.

## 6. Outcomes / Economic Value Analysis

This is the most important section of this report, and the most likely thing to be misquoted if this project is summarized casually — so it is stated plainly here.

A full, symmetric approval/return frontier was built (both policies swept across a full range of target approval rates, calibrated on train, evaluated on test):

- **Champion** has a genuine interior optimum: RAROC peaks at **2.5%** at a **54.5%** approval rate (FICO cutoff 682) and gets worse in either direction from there — a well-behaved, credible tradeoff curve.
- **Challenger's** RAROC gets progressively **worse** the more selective it becomes, across the *entire* tested range (50–99% approval). Its best point sits at the loosest edge of that range: **-0.5%** RAROC at **98.4%** approval — not a genuine peak, an edge effect.
- This pattern held (barely changed: -1.9% → -0.5%) even after removing interest rate as a challenger feature — ruling out the initial hypothesis that this was purely a rate/revenue confound (Section 7.4).

**Conclusion, stated as plainly as possible: a modestly predictive PD model (AUC 0.64) does not automatically translate into a positive RAROC improvement over a simple rule, once revenue mix and realistic cost assumptions are properly accounted for.** The model's risk ranking is real — tightening challenger's threshold does reduce the realized loss rate — but the loans it identifies as "riskiest" are not risky enough, relative to the revenue given up, to make declining them RAROC-positive under the current cost assumptions. This is a genuine, examined finding, not an artifact of a bug. It is also the single most important thing for anyone reading this project to understand before drawing any conclusion about the challenger's value.

## 7. Identified Issues & Remediation Log

A real model-risk function tracks every issue found and how it was resolved. This project's issue log, in full:

| # | Issue found | How found | Resolution | PR |
|---|---|---|---|---|
| 1 | Reject inference: fixed FICO cutoff non-binding (100% approval) | Running against real data | Volume-matched, swap-set calibration | 2.1 |
| 2 | RAROC math: revenue used a flat multiplier instead of real amortization | Reviewing the RAROC formula against real loan economics | Real fixed-payment amortization formula | 2.2 |
| 3 | RAROC > 100% (201.5%/154.3%) — a genuine bug | Running against real data | Annualized revenue/loss to match the capital charge's time basis | 2.3 |
| 4 | Volume-matching ≠ RAROC-optimal (challenger had lower loss rate but worse RAROC) | Running against real data, checking the counterintuitive result rather than accepting it | Built a RAROC-optimized threshold sweep alongside volume-matching | 2.4 |
| 5 | Vintage curve needed real time-to-default data, which the schema didn't have | Reviewing PR 3's design against the actual schema before building it | Added `last_pymnt_d` and derived `months_on_book` | 2.5 |
| 6 | Champion's frontier was never swept — only challenger's | Reviewing PR 2.4's asymmetry | Built a symmetric frontier for both policies | 3 |
| 7 | Interest rate included as a challenger feature — a confound with RAROC's revenue side | Reviewing PR 3's frontier result before trusting it | Removed rate as a model input (pre-pricing risk model) | 3.1 |

Seven real issues, each found by checking results against real data rather than accepting a plausible-looking number, and each documented with its actual resolution. This log is itself evidence of the kind of process discipline SR 11-7 asks for — independent challenge, not just a single pass of development.

## 8. Fair Lending Review

**Not yet built.** Lending Club's data has no protected-class field. Any fair-lending screen built on this data will need to use either a fully synthetic proxy (clearly labeled as illustrative-only) or a BISG-style (Bayesian Improved Surname Geocoding) proxy consistent with real regulatory practice — but even BISG here would be illustrative, not evidentiary, given the data's limitations. Scoped as a near-term next step, not yet complete.

## 9. Assumptions Inventory

| Assumption | Value | Status |
|---|---|---|
| Loss Given Default (LGD) | 55% | Illustrative, not calibrated to any real institution |
| Annual operating cost rate | 5% of loan amount | Illustrative |
| Annual capital charge rate | 8% of loan amount | Illustrative |
| Champion FICO cutoff (reference constant) | 660 | Superseded by volume-matched/frontier calibration in practice |
| Target approval rate (where used as a fixed default) | 85% | A default, not a business-validated target |

A real deployment requires every row in this table to come from Regional Finance's actual Treasury, servicing, and capital planning functions — none of these should be treated as findings about any real company's economics.

## 10. Scope Boundaries / What This Model Cannot Tell You

- Anything about Regional Finance's actual portfolio, borrowers, or economics — this is Lending Club's book.
- Anything about applicants who would have been declined by Lending Club — reject inference means this population is pre-screened.
- Real fair-lending exposure — no real demographic data exists in this project.
- Whether a *stronger* model (richer features, non-linear methods) would find genuine RAROC-positive selectivity where this one didn't — this report evaluates the model that was built, not the ceiling of what's possible.

## 11. Validation Verdict

**Not approved for production use, and not intended to be presented as such.** As a methodology demonstration — champion/challenger design, time-based evaluation discipline, RAROC-over-accuracy framing, and a real, traceable issue-remediation log — this project is sound and worth standing behind. As a claim about any real institution's risk-return profile, it is not, and every artifact in this repo says so explicitly.

**Conditions for any future production consideration**:
1. Retrain and re-evaluate entirely on the institution's own loan-level data.
2. Add bureau-depth features beyond what a single loan-origination file provides.
3. Replace every assumption in Section 9 with the institution's actual Treasury/servicing/capital figures.
4. Independent validation by a reviewer who did not build the model (this report is a self-applied version of that discipline, not a substitute for it).
5. An ongoing monitoring plan (population stability, performance drift) — not yet built here.

---

*This report is itself a portfolio artifact demonstrating the model-validation discipline described in SR 11-7 (independent review, documented conceptual soundness assessment, outcomes analysis, and an explicit approval/non-approval verdict) — applied to this project's own model, not received from an external validator.*
