# Path to Production

**Purpose**: this project is a portfolio/methodology artifact, built on public Lending Club
data (see `CLAUDE.md`, `docs/MODEL_VALIDATION.md`). This document states plainly what would
need to change, stage by stage, if Regional Finance (or any real institution) handed over its
actual loan tape and asked for this pipeline to be adapted into a real decision-support tool.
Nothing here is a claim that this project is close to production — it's the opposite: a
precise map of the gap, stage by stage, so the gap is never ambiguous.

**The core claim of this document**: the *pipeline architecture* (ingest → clean → engineer
features → train/evaluate → economic analysis → fairness screen → monitor) is genuinely
reusable. The *specific numbers this pipeline currently produces* are not — they're artifacts
of Lending Club's book, Lending Club's platform economics, and a set of stated, illustrative
assumptions. Swapping the data source does not mean starting over; it means re-running an
already-built, already-tested pipeline against different inputs and re-validating every output.

---

## Stage-by-stage: what changes, what stays

### 1. Data ingestion (`src/data_loader.py`)

**Stays**: the ingestion pattern — schema validation, structured JSONL run logging at every
filter step, explicit scope decisions logged and stated (not silent), a data-quality filter
layer, and tests against a synthetic fixture so the logic is verified independent of any real
file being present.

**Changes**: the actual source. Replace the Lending Club CSV loader with a connector to the
institution's own loan origination system / data warehouse. Every filter currently applied
(grade C–F scope, matured-outcome-only scope, platform-maturity year cutoff) needs to be
re-examined against the real institution's actual population — these were reasonable choices
*for Lending Club's dataset specifically*, not universal rules.

**New requirement not present in this project**: a real institution's own loan tape will
almost certainly include declined applications, not just accepted ones — which removes the
reject-inference limitation documented throughout this project (`CLAUDE.md`,
`docs/MODEL_VALIDATION.md` Section 4) and allows a genuinely bindable underwriting cutoff to
be evaluated, not just a volume-matched proxy for one.

### 2. Feature engineering (`src/features.py`)

**Stays**: the time-based train/test split discipline (mandatory, not optional — loan
performance and applicant mix drift over time, a random split leaks the future into training).

**Changes**: the feature set. This project's challenger model uses only what a single Lending
Club loan-origination export provides. A real deployment needs bureau-depth features —
tradeline history, full inquiry history, collections detail — which materially improve
discriminatory power beyond this project's modest AUC 0.64 (`docs/MODEL_VALIDATION.md`
Section 5).

**Carries forward as a real methodological decision, not a new one**: the project's own
finding that pricing (`int_rate_frac`) should NOT be a model input for an approve/decline
decision (PR 3.1) — a real deployment should keep risk-scoring and pricing as separate,
sequential decisions, exactly as real underwriting practice does.

### 3. Modeling (`src/models.py`)

**Stays**: champion-as-current-practice vs. challenger-as-proposed-improvement framing (the
real, industry-standard meaning of the term, not two competing ML models racing each other —
see `README.md`'s PR 2 rationale); RAROC computed from realized outcomes, not either policy's
self-assessed risk; the volume-matched AND RAROC-optimized dual comparison, since this
project found the two are not interchangeable (PR 2.4).

**Changes**: model complexity. A real production PD model would likely go beyond logistic
regression — but this project deliberately did not, to keep the champion/challenger
comparison an honest "rules vs. a real statistical alternative" story rather than duplicating
Credit-Risk-Monitor's existing LR-vs-XGBoost comparison. A real deployment should evaluate
whether a more complex model actually earns its complexity via the same rigor this project
applied — real economic backtesting, not just AUC.

### 4. Economics (`LGD`, `OPEX_RATE`, `CAPITAL_RATE` in `src/models.py`)

**Changes entirely.** These three constants are explicitly documented as illustrative, not
calibrated to any real institution (`docs/MODEL_VALIDATION.md` Section 9). The sensitivity
analysis (`src/sensitivity.py`) already proved the champion-vs-challenger conclusion is robust
across a *plausible range* of these — but "plausible" was still a guess. A real deployment
replaces every one of these three numbers with the institution's actual Treasury-provided
funding cost, actual collections/recovery-curve data (for a real LGD, not a flat assumption),
and actual servicing cost data (for a real opex rate). This is Treasury/Finance's work, not a
data science task — and it's the single highest-priority number to get right before trusting
any RAROC output from this pipeline on real data.

### 5. Vintage & frontier analysis (`src/vintage.py`, `src/frontier.py`)

**Stays as-is, structurally.** Both are already built on real time-to-default data
(`months_on_book`, derived from `last_pymnt_d` — PR 2.5) and a symmetric sweep across both
policies (PR 3) — this is genuinely reusable methodology, not something that needs rework.

**Changes**: nothing structural. Re-running these against a real institution's own loan tape
is a direct swap of the input dataframe — the logic doesn't need to change.

### 6. Fair lending (`src/fair_lending.py`)

**Changes entirely — this is the largest real gap in the whole pipeline.** This project's
screen uses a fabricated, illustrative-only state probability table specifically because
Lending Club's data has no protected-class field and no borrower name (see `CLAUDE.md`,
`docs/MODEL_VALIDATION.md` Section 8). A real deployment requires:
- Real Census Bureau surname-race distribution data and real block-group/ZIP demographic
  data, for a genuine two-leg BISG proxy (not the single-leg, geography-only version here).
- Routing through the institution's actual fair-lending compliance function — protected-class
  proxy data of this kind should never be generated or used ad hoc by a model-building team;
  it needs the same governance real BISG analyses get in practice.
- Legal/compliance review before any output from this screen is used to inform an actual
  policy decision — this is categorically different from every other stage of this pipeline,
  which can be validated by a technical review alone.

**Stays**: the four-fifths mechanism itself, and the pattern of running the parity screen
across every policy variant (champion, volume-matched challenger, RAROC-optimized
challenger) rather than just one — that structure is correct and reusable.

### 7. Ongoing monitoring — built (`src/monitor.py`), on a real proxy for "live"

**Built, using a real proxy for live monitoring.** This project has no actual production
stream, so `src/monitor.py` demonstrates PSI drift monitoring on the closest real substitute
available: the time-based train (pre-2015) vs. test (2015+) split already used throughout
this project. This is a genuine before/after population comparison on real data, not a
synthetic demonstration — the same pattern proven in Credit-Risk-Monitor's `monitor.py`.

**What changes for a real deployment**: swap the "test" population for a genuinely live,
rolling window of recent applicants, scored on a recurring cadence (daily/weekly/monthly)
against the original training population as the fixed baseline — the PSI math itself doesn't
change, only the cadence and what counts as "the population being checked."

---

## Governance requirements, not just code changes

A real production consideration needs, at minimum:
1. **Independent model validation** — by a reviewer who did not build the model.
   `docs/MODEL_VALIDATION.md` is a self-applied version of this discipline; it is explicitly
   not a substitute for a real independent review.
2. **Treasury/Finance sign-off** on every economic assumption (Section 4 above) before any
   RAROC output is trusted.
3. **Compliance/legal ownership** of the fair-lending function (Section 6 above) — this
   cannot be a data-science-team-only deliverable in a real deployment.
4. **A defined re-validation cadence** — SR 11-7-consistent practice requires periodic
   re-validation, not a one-time approval.

## What this document does NOT claim

- That this pipeline is close to production-ready as-is — see `docs/MODEL_VALIDATION.md`'s
  explicit non-approval verdict.
- That swapping in real data guarantees the same conclusion (champion beats challenger) would
  hold — that would need to be re-derived, not assumed, on the real population.
- That this document itself constitutes a real implementation plan with timelines/resourcing
  — it is a technical gap map, not a project plan.

---

*This document is itself part of the portfolio demonstration: showing the ability to
distinguish reusable methodology from dataset-specific findings, and to be precise about
where a technical build ends and organizational/governance work begins.*
