# Engineering Virality with Bayesian Optimization

**Primary study:** [Influence Maximization using Bayesian Optimization on the Higgs Graph](https://app.chinmayarora.com/blog/influence-maximization-bayesian-optimization-higgs/)

Code behind the figures in
[Engineering Virality with Bayesian Optimization](https://app.chinmayarora.com/blog/virality-bo/)

> **Correction (Nov 2026).** Earlier versions of this code built the graph
> with edges pointing from retweeter to retweeted, so every cascade ran *against*
> the direction information actually flows. "Top-degree" meant "retweeted the
> most", not "was retweeted the most". The bug is fixed and every number below
> comes from a full rerun. **The earlier headline (BO beats top-degree by
> +2.5% / +4.5%) does not survive the fix.** The published posts still quote the old
> numbers and need updating. See [Changes in the rerun](#changes-in-the-rerun).

Influence maximization under the Independent Cascade (IC) model on the SNAP
Higgs **retweet** network (largest weakly-connected component: 223,833 nodes /
308,596 edges, oriented retweeted → retweeter). The question: can Bayesian
Optimization select a seed set of k=20 nodes that spreads further than a
top-degree heuristic?

All numbers: [`RESULTS.md`](RESULTS.md) (generated from the pipeline outputs, not
typed by hand) and `results/summary.json`.

## The four-stage experiment

Every stage uses the **same objective** (k=20, IC p=0.12, 80 simulations per
evaluation) and the **same top-degree baseline** (global top-20 by number of
retweeters, scored at 4,000 simulations: **8,012.5**). BO and random search get
the **same number of evaluations** in the same space. Each run's best point is
re-evaluated with 400 fresh simulations, and the table reports mean ± sd across runs.

| Stage | Script | Search space | BO (mean ± sd over runs) | vs top-degree | Random search, same budget | Best BO set ∩ top-degree |
|---|---|---|---|---|---|---|
| 1. Rule-tuning | `01_rule_tuning_bo.py` | 3-D blend of degree / PageRank / k-core | 8,009 ± 15 (5 runs) | −0.04% | 7,998 ± 13 | 20/20 |
| 2. Rules + redundancy | `02_stronger_variant.py` | 4-D: exponents, redundancy penalty, hop | 8,007 ± 9 (5 runs) | −0.06% | 8,002 ± 22 | 20/20 |
| 3. Free-node, naive | `03_freenode_naive.py` | 120-D spectral-embedding anchors | 737 ± 431 (4 runs) | −91% | 748 ± 418 | 1/20 |
| 4. Free-node, constrained | `04_freenode_constrained.py` | 13-D (12 latent + diversity weight) over top-400 pool | 7,919 ± 41 (5 runs) | −1.2% | 7,492 ± 204 | 19/20 |

Controls:
- **CELF** (lazy greedy on 200 fixed live-edge snapshots, top-3000 pool) picks
  **the same 20 seeds as top-degree** (8,003.5 fresh; difference is noise).
- **Decoder-only** (stage 4 decoder with no BO bias, diversity weight swept on a
  grid) also returns exactly the top-degree set.

## What the rerun supports

1. **On this graph and IC setting, top-degree is effectively optimal for k=20.**
   CELF, the standard near-optimal greedy, picks the same set. Cascades are
   shallow (retweeters are rarely retweeted themselves), and the top-20 accounts'
   direct audiences overlap only ~13%. That leaves very little redundancy for
   diversity-seeking strategies to recover.
2. **BO never beats top-degree here.** In stages 1–2 it rediscovers the top-degree
   set exactly. In stage 4 it lands one seed away and is 1.2% worse.
3. **Search-space design still decides whether BO can *match* the baseline.**
   In low-dimensional, degree-anchored spaces (1, 2, 4) BO gets within ~1%. In
   the naive 120-D space (3), BO and random search both collapse to ~9% of the
   baseline.
4. **BO beats random search only where the space is hard.** In stage 4, BO
   averages 7,919 ± 41 versus 7,492 ± 204 for random search, with lower variance
   and every BO run above every random-search run. In stages 1–2 both methods sit
   on the same plateau. In stage 3 neither makes progress.

What it does **not** support: any claim that BO, diversity or redundancy
penalties improve on top-degree for this graph.

## Methods

- **Objective:** mean activated count over Monte Carlo IC simulations (noisy black box).
- **Optimizer:** GP regression (Matérn 2.5 + white noise) with Expected Improvement.
- **Matched curves:** `curve[i]` = best value after `i+1` objective evaluations,
  for BO (initial design included) and random search alike.
- **Run-to-run variability:** each run's best re-evaluated fresh. Reported as
  mean ± sd and as "runs above top-degree".
- **Set-level tests:** Welch t-tests over 30×150-simulation estimates. These
  compare two *fixed* seed sets and support "this set beats or loses to top-degree",
  not "the method beats top-degree".
- **CELF:** sample-average approximation over fixed live-edge snapshots, which keeps
  the objective submodular so lazy evaluation is valid. The chosen set is scored with
  independent simulations.

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r ../requirements.txt
# get the data - see data/README.md - and place the retweet edgelist in pipeline/
cd pipeline
python 00_build_graph.py            # -> rt_graph.pkl (edges reversed to diffusion direction)
python 01_rule_tuning_bo.py         # -> baselines.pkl, bo_results.pkl
python 02_stronger_variant.py       # -> bo_strong_results.pkl
python 03_freenode_naive.py         # -> bo_freenode_results.pkl
python 04_freenode_constrained.py   # -> bo_constrained_results.pkl
python 05_summary.py                # -> ../RESULTS.md, ../results/summary.json
cd ../charts && for f in 0*.py; do python $f; done   # -> charts/figures/*.png
```

The full pipeline takes about 3 minutes on a laptop. `QUICK=1` runs a smoke test with
tiny budgets, and `05_summary.py` refuses to summarise those outputs. `IC_P=…`
reruns everything at another activation probability. Shared settings live in
`pipeline/common.py`. Loading a graph pickle built with the old edge direction
fails with an error.

## Changes in the rerun

Fixes made after a code audit (Nov 2026):

| Issue | Fix |
|---|---|
| Edges added retweeter → retweeted, so cascades ran backwards | `00_build_graph.py` adds retweeted → retweeter. The graph pickle carries a direction tag that is checked on load. PageRank is computed on the endorsement graph. |
| BO curve point *i* covered *i*+5/6/12 evaluations, random-search point *i* covered *i*+1 | Both curves are indexed by total evaluations (`common.run_bo` / `run_random`) |
| Stage 1 used IC p=0.05 while stages 2–4 used 0.12. "179" was mis-attributed. Charts hardcoded `[179, 91, 183.1]` | One objective for all stages. Charts read only from pickles and assert that all stages agree. |
| Stage 4's top-degree came from a different pool than stage 3's | One shared global top-degree baseline (`baselines.pkl`) |
| Stage 4 had no random-search control and was described as "12-D" (actually 13-D) | Added a 13-D random-search control with the same budget, a decoder-only control, and 5 runs re-evaluated individually |
| p-values presented as evidence that BO beats top-degree | Relabelled as set-level. Run-level mean ± sd reported. |
| CELF used 20/40-sim noisy marginal gains, which breaks lazy greedy | Snapshot (SAA) CELF, scored with independent simulations |
| Stage 4 saved no pickle, stage 1 had a dead `vals` line, and charts hardcoded values and `/home/claude` paths | Fixed |

## Honest limitations

- Results are specific to this graph, k=20, and IC with uniform p=0.12. A
  weighted-cascade model, a larger k, or the follower graph (deeper cascades,
  more overlap) could change the picture. None of these were tested.
- The stage-3 collapse is partly a budget artifact (120-D, 60 noisy evaluations).
- Run counts are small (4–5 per stage), so the run-level sds are rough.
