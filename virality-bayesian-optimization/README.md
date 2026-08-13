# Engineering Virality with Bayesian Optimization

Code behind the figures in
[Engineering Virality with Bayesian Optimization](https://app.chinmayarora.com/blog/virality-bo/)

Influence maximization under the Independent Cascade (IC) model on the SNAP
Higgs **retweet** network (largest weakly-connected component: 223,833 nodes /
308,596 edges). The question: can Bayesian Optimization select a seed set of
k=20 nodes that spreads further than a top-degree heuristic?

The answer turns out to depend entirely on **how the search space is defined.**

## The five-stage experiment

| Stage | Script | Design space | Result |
|---|---|---|---|
| 1. Rule-tuning | `01_rule_tuning_bo.py` | BO reweights degree/PageRank/k-core rankings | Matches/edges top-degree (179 vs 175); 18/20 seeds shared — refinement, not diversity |
| 2. Stronger variant | `02_stronger_variant.py` | + redundancy penalty, CELF baseline, higher IC p | Beats top-degree +2.5% (p<0.0001), but gain is from exponent tuning, not the diversity mechanism |
| 3. Free-node, naive | `03_freenode_naive.py` | BO picks any nodes via a 120-D spectral embedding | **Collapses to 91 (−48%).** Maximal diversity (1/20 shared), minimal spread — unconstrained search starves at this budget |
| 4. Free-node, constrained | `04_freenode_constrained.py` | BO picks diverse *subsets of strong nodes* (12-D) | **Best result: 183 (+4.5%, p<0.0001), 14/20 shared.** Diversity finally pays when anchored to high-influence nodes |
| 5. Approach B: persona-conditioned IC | `05_persona_conditioned_ic.py` | BO balances topology, topic affinity, sharing propensity, and redundancy under heterogeneous activation probabilities | **Implemented as a sensitivity-tested extension.** No Higgs performance claim is reported until the persona-to-node conditioning grid is run |

**Headline finding:** same graph, same budget, same objective — three framings
produce 179 / 91 / 183. The design space, not the optimizer, decides whether
BO helps. This is graph-agnostic; the absolute margins are modest and specific
to this graph and IC p.

## Approach B — why persona conditioning matters

Stages 1–4 use a single activation probability for every eligible edge. That is
a standard simplifying assumption, but it makes behavior homogeneous even
though the Higgs topology is real. Approach B replaces the global probability
with a topic-conditional function of the receiving persona's content affinity,
sharing propensity, and affinity similarity to the sender.

The first two content profiles are deliberately contrastive:

- an ML paper, using fields such as machine-learning familiarity, technology
  and science interest, technical literacy, need for cognition, and reading
  preference;
- a sports clip, using sports interest and exposure, video preference,
  excitement seeking, and social-media attitude.

BO then searches an interpretable four-weight seed rule: graph strength, topic
affinity, sharing propensity, and redundancy. The optimization problem is no
longer just "who is structurally influential?" It becomes "which strong nodes
are behaviorally plausible first adopters for this content type?"

### The conditioning gate comes first

MatrAIx exposes 1,290 categorical persona fields, but it does not provide an
identity join to the anonymized Higgs nodes. Its public release is also explicit
that marginal calibration does not guarantee a representative joint
distribution. A random global persona sample therefore cannot be treated as a
model of Twitter users during the July 2012 Higgs event.

`05_persona_conditioned_ic.py audit` checks schema coverage, age eligibility,
source composition, effective sample size after soft platform-fit weighting,
and separation between the ML and sports affinity signals. The node assignment
is then rerun over a required topology/persona coupling grid of `0.00`, `0.35`,
and `0.70`. A result that disappears across that grid is assumption-sensitive,
not robust.

See [`docs/approach-b-persona-conditioned-ic.md`](docs/approach-b-persona-conditioned-ic.md)
for the probability model and reporting contract.

## Three-tier provenance

Approach B must be described as:

1. **Real network topology:** the SNAP Higgs retweet graph.
2. **Synthetic node attributes and behavior:** MatrAIx-derived personas are
   assigned to anonymous nodes, and the activation coefficients are assumed.
   Even a human-grounded MatrAIx row becomes synthetic metadata after that
   unobserved join.
3. **Real method:** heterogeneous Independent Cascade simulation, Monte Carlo
   evaluation, and Gaussian-process Bayesian Optimization.

The MatrAIx release itself contains grounded and synthetic persona records. That
does not make the assigned Higgs-user attributes observed data.

## Methods

- **Objective:** mean activated count over Monte Carlo IC simulations (a noisy
  black-box — well suited to BO).
- **Optimizer:** Gaussian-Process regression (Matérn 2.5 + white-noise kernel)
  with Expected Improvement acquisition.
- **Baselines:** top-degree, top-PageRank, top-k-core, random seeds, and CELF
  (lazy-greedy influence maximization).
- **Validation:** winning seed sets re-evaluated at high precision (400 sims);
  BO-vs-baseline gaps confirmed with two-sample t-tests over 30 estimates.

## Reproducing

```bash
pip install -r ../requirements.txt
# 1. get the data — see data/README.md, then place the retweet edgelist in pipeline/
cd pipeline
python 00_build_graph.py            # -> rt_graph.pkl
python 01_rule_tuning_bo.py         # -> bo_results.pkl
python 02_stronger_variant.py       # -> bo_strong_results.pkl
python 03_freenode_naive.py         # -> bo_freenode_results.pkl
python 04_freenode_constrained.py   # prints final comparison
# Approach B starts with the persona suitability gate
python 05_persona_conditioned_ic.py audit \
  --personas ../data/matraix/matraix-persona-sample.parquet
# Run only after rt_graph.pkl exists; all coupling values are required
python 05_persona_conditioned_ic.py experiment \
  --personas ../data/matraix/matraix-persona-sample.parquet \
  --graph rt_graph.pkl --couplings 0 0.35 0.70
# then render figures
cd ../charts && python 03_freenode_charts.py   # etc.
```

Note: the chart scripts in `charts/` read the result pickles produced by the
pipeline and write PNGs to a local `figures/` directory. Random seeds are
fixed where it matters, but IC simulation is stochastic — exact numbers will
vary by a fraction of a percent run to run.

`charts/04_graphbo_charts.py` is a companion ("BO meets graphs") that renders
two extra figures directly from `rt_graph.pkl`: a node-scatter showing where
the top-degree vs. diverse-subset strategies place their seeds, and a
combinatorial blow-up illustrating why the seed-selection space (~10^88 sets
at k=20) cannot be brute-forced. It needs only the cached graph, not the BO
result pickles.

## Honest limitations

- Results are specific to this graph and IC activation probability (p=0.12 in
  stages 2–4).
- Approach B's behavioral probabilities are synthetic, not fitted to observed
  Higgs retweet opportunities. Its persona-to-node assignment is unidentified
  and must be treated as a sensitivity parameter.
- The public MatrAIx sample is suitable for a schema/method smoke test, not a
  claim about the population of Twitter in 2012.
- Absolute margins are small (a few percent). The structural finding about
  search-space design is the contribution, not the magnitude.
- The naive free-node failure (stage 3) is partly a budget artifact: 120-D
  search with ~50 noisy evaluations. More evaluations or a better embedding
  might recover some performance — untested.
