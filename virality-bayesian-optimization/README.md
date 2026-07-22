# Engineering Virality with Bayesian Optimization

**Primary study:** [Influence Maximization using Bayesian Optimization on the Higgs Graph](https://app.chinmayarora.com/blog/influence-maximization-bayesian-optimization-higgs/)

Code behind the figures in
[Engineering Virality with Bayesian Optimization](https://app.chinmayarora.com/blog/virality-bo/)

Influence maximization under the Independent Cascade (IC) model on the SNAP
Higgs **retweet** network (largest weakly-connected component: 223,833 nodes /
308,596 edges). The question: can Bayesian Optimization select a seed set of
k=20 nodes that spreads further than a top-degree heuristic?

The answer turns out to depend entirely on **how the search space is defined.**

## The four-stage experiment

| Stage | Script | Design space | Result |
|---|---|---|---|
| 1. Rule-tuning | `01_rule_tuning_bo.py` | BO reweights degree/PageRank/k-core rankings | Matches/edges top-degree (179 vs 175); 18/20 seeds shared — refinement, not diversity |
| 2. Stronger variant | `02_stronger_variant.py` | + redundancy penalty, CELF baseline, higher IC p | Beats top-degree +2.5% (p<0.0001), but gain is from exponent tuning, not the diversity mechanism |
| 3. Free-node, naive | `03_freenode_naive.py` | BO picks any nodes via a 120-D spectral embedding | **Collapses to 91 (−48%).** Maximal diversity (1/20 shared), minimal spread — unconstrained search starves at this budget |
| 4. Free-node, constrained | `04_freenode_constrained.py` | BO picks diverse *subsets of strong nodes* (12-D) | **Best result: 183 (+4.5%, p<0.0001), 14/20 shared.** Diversity finally pays when anchored to high-influence nodes |

**Headline finding:** same graph, same budget, same objective — three framings
produce 179 / 91 / 183. The design space, not the optimizer, decides whether
BO helps. This is graph-agnostic; the absolute margins are modest and specific
to this graph and IC p.

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
- Absolute margins are small (a few percent). The structural finding about
  search-space design is the contribution, not the magnitude.
- The naive free-node failure (stage 3) is partly a budget artifact: 120-D
  search with ~50 noisy evaluations. More evaluations or a better embedding
  might recover some performance — untested.
