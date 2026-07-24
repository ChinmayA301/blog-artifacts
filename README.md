# blog-artifacts

Reproducible code behind the data-driven figures on
[app.chinmayarora.com/blog](https://app.chinmayarora.com/blog/).

Each directory maps to one blog post and is self-contained: the experiment
code, the figure-generation code, and a README explaining what each figure
shows and how to reproduce it. Most figures live in the blog's own assets;
studies explicitly packaged as self-contained audits may also keep checked
figures alongside their data and documentation.

## Contents

### `forecast-credibility/`
→ post: [Where Do the AI Numbers Actually Come From?](https://app.chinmayarora.com/blog/ai-forecast-numbers/)

Stress-tests the Goldman Sachs SpaceX $3.2B → $322B (2030) AI-revenue
projection against its own filings: an implied-CAGR waterfall, a
front/back-loaded path comparison, and a Monte Carlo of terminal revenue
under correlated assumptions.

### `virality-bayesian-optimization/`
→ post: [Engineering Virality with Bayesian Optimization](https://app.chinmayarora.com/blog/virality-bo/)

Influence-maximization on the SNAP Higgs retweet graph (223,833-node WCC).
A four-stage experiment showing that **the search-space design — not the
optimizer — decides whether Bayesian Optimization beats a top-degree
heuristic.** Includes a real Gaussian-Process BO loop, Independent Cascade
simulation, CELF and centrality baselines, significance testing, and a
"BO meets graphs" companion visual set.

### `signalgraph-fake-stars/`
→ post: SignalGraph — detecting inorganic GitHub stars

A robust velocity-anomaly detector (median/MAD z-score + burst-clustering)
that separates organic star growth from purchased bursts, and a
credibility-adjusted ranking that reshuffles a gamed leaderboard.

### `ai-human-detector/`
→ post: AI and human imperfection

A logistic detector on text "imperfection" features, evaluated honestly with
ROC, calibration, and an explicit look at the overlap zone where it fails.

### `media-influence-causality/`
→ post: American media and global power

A lead/lag cross-correlation paired with a causal DAG, built to show why a
clean lagged correlation still does not settle a causal claim.

### `football-pressing-context/`
→ post: Pressing is not a morality stat

An exploratory StatsBomb-open-data study of why raw forward pressure volume is
an incomplete football metric. It builds player/team samples from public event
data, then compares a naive pressure-only model against attacking-tradeoff and
team-context models.

### `ai-value-translation-audit/`
→ study: AI Value Translation Audit

A secondary-data audit of nine documented enterprise-AI use cases. It scores
workflow specificity, data readiness, evaluation, accountability, and value
translation to distinguish local productivity gains from governed operational
performance.

### `metric-substitution-audit/`
→ post: [The Proxy Gap: When Access Stops Measuring Capability](https://app.chinmayarora.com/blog/metric-substitution-audit/)

A source-audited study of metric substitution. It compares rural school
enrollment with foundational reading across all 27 ASER 2024 state/UT pages,
then uses official Enforcement Directorate aggregates to demonstrate why
changing a denominator changes the question being answered. The folder keeps
the code, extracted CSVs, methods, source audit, machine-readable results, and
checked figures together.

## A note on data provenance

The projects are separated by evidence type, and each subproject's README says
which type applies:

- **Real data, real findings:** `forecast-credibility` (public filings) and
  `virality-bayesian-optimization` (SNAP Higgs graph).
- **Real public data, scoped exploratory findings:** `football-pressing-context`
  (StatsBomb open event data; incomplete competition coverage, no tracking).
- **Real reported outcomes, author-scored synthesis:**
  `ai-value-translation-audit` (nine linked papers and case studies; a small,
  purposive sample scored with an explicit rubric, not a causal meta-analysis).
- **Real public aggregates, descriptive cross-sectional audit:**
  `metric-substitution-audit` (ASER 2024 state tables and official Enforcement
  Directorate pipeline totals; no individual-level or party-coded causal
  inference).
- **Real method, synthetic-but-labeled data:** `signalgraph-fake-stars`,
  `ai-human-detector`, and `media-influence-causality`. The live sources
  (authenticated GitHub API; paired human/AI corpora; assembled influence
  indices) were not reachable when the figures were produced, so these
  demonstrate the *method* on labeled synthetic data. Each README documents the
  exact path to promote it to a real finding.

## Reproducing

Each subproject has its own README with exact steps. In general:

```bash
pip install -r requirements.txt
```

The forecast figures run standalone. The BO experiments need the Higgs
retweet edgelist (see `virality-bayesian-optimization/data/README.md`),
after which `00_build_graph.py` produces the cached graph and the numbered
pipeline scripts run in order.

## Note on results

The BO experiments report honest outcomes, including a decisive negative
result (naive free-node selection underperforms top-degree by ~48%). The
value of the work is the structural finding about search-space design, which
is graph-agnostic; the absolute margins are modest and specific to this graph
and cascade probability.
