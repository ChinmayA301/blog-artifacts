# Validation: does the simulator match real diffusion?

The influence-maximization experiments compare seeding strategies inside a
cascade simulator, so the comparison is only as good as the simulator. This folder
tests the simulator against what happened on Twitter during the Higgs announcement
week, using two more files from the same SNAP release:

- `higgs-social_network.edgelist.gz`: who follows whom (456k users, 14.9M edges);
- `higgs-activity_time.txt.gz`: timestamped retweets, mentions and replies, 1–7 July 2012.

Run `../pipeline/00b_build_follower_graph.py`, then `python calibrate_and_holdout.py`
and `python account_ranking_check.py`. Outputs: `calibration.json`,
`holdout_results.json`, `holdout_check.png`, `account_ranking_check.json`.

## 1. The retweet graph is the wrong substrate

The original study simulated cascades over the retweet graph. Every edge in that
graph is a retweet that already happened, so it spreads simulated diffusion along
realised outcomes. Exposure on Twitter runs along follow edges.

- **59%** of unique retweets have the retweeter following the source, against 13% for the
  reverse direction. The follower graph explains most diffusion; the rest arrives
  through search, trends and other routes.
- **5.9%** of retweeters were themselves retweeted. Real cascades are shallow.
- The top-20 accounts' audiences overlap **13%** on the retweet graph but **44–65%** on
  the follower graph.

## 2. Calibration

Activation probabilities are estimated from follow-explained adoptions (a follower
retweeting someone they follow) against the model's expected exposure, **over every
user active in the window**. Counting only users who were retweeted at least once
conditions on success and inflates p (0.024 instead of 0.015).

Two models:
- **uniform IC**: p_uv = p;
- **weighted cascade**: p_uv = c / max(followees(v), F). Attention is split across
  everyone v follows; F is a floor on the denominator (section 4).

| Window | Active users | Uniform p | Weighted cascade c (F = 20) |
|---|---|---|---|
| 1–3 July (before the announcement) | 35,566 | 0.0050 | 0.232 |
| 4–7 July (held-out window) | 287,476 | 0.0140 | 0.679 |
| Whole week (used for the optimisation rerun) | 304,474 | 0.0149 | 0.726 |

Transmissibility roughly **tripled** at the 4 July announcement.

## 3. Held-out test

Each model is seeded with the held-out window's 22,991 originators (users retweeted
by a follower who did not themselves retweet anyone in that window) and compared
with the 139,792 follow-explained adopters observed.

| Model, parameters from | Predicted adopters | vs observed | Cascade depth (observed 0.056) |
|---|---|---|---|
| Uniform IC, 1–3 July | 30,489 | 0.22× | 0.123 |
| Weighted cascade F=20, 1–3 July | 29,414 | 0.21× | **0.053** |
| Uniform IC, held-out window itself | 90,276 | 0.65× | 0.167 |
| Weighted cascade F=1, held-out window itself | 97,945 | 0.70× | 0.088 |
| Weighted cascade F=20, held-out window itself | 94,219 | 0.67× | 0.118 |

- **Out of sample, every model under-predicts spread by about 4.7×.** A model calibrated
  before the announcement does not transfer to it: the event changed how readily people
  retweeted. Forward prediction needs regime-aware calibration.
- **Calibrated in-window, models recover two-thirds of spread.** Weighted cascade fits
  cascade depth better than uniform IC, which more than doubles it.

## 4. The crawl artifact, and why aggregate fit cannot see it

The SNAP crawl records follows only among Higgs-involved users, so followee counts are
truncated. **33,162 users (7.3%)** appear to follow exactly one account, and half of them
follow **node 1**. With no floor (F = 1), weighted cascade gives each of those users
p ≈ 0.49:

| | Node 1 |
|---|---|
| Followers | 16,280 |
| Observed retweets all week | **0** |
| Predicted singleton reach, F = 1 | 12,403 |
| Predicted singleton reach, F = 20 | 1,044 |

Node 1 is never an active seed, so the aggregate held-out test cannot catch this.
Optimisers find it immediately: with F = 1, CELF and rule-tuning BO "beat" top-degree by
+45% and +43%, almost entirely by adding node 1.

**The floor is chosen by out-of-sample fit**, calibrating on 1–3 July and testing from
4 July. That picks F = 20, whose depth (0.053) is closest to observed (0.056). The choice
was fixed before the optimisation was rerun.

## 5. Per-account check (`account_ranking_check.py`)

Seed selection is a ranking problem, so the decisive test is whether the model ranks
candidate accounts the way reality does. For the top-3,000 accounts by followers, it
compares the model's expected direct adoptions with the observed number of distinct
follower-retweeters.

| Model | Spearman ρ | Model top-20 in observed top-20 |
|---|---|---|
| Uniform IC | 0.45 | 8/20 |
| Weighted cascade F=1 | 0.50 | 8/20 (node 1 predicted 8,010, observed 0) |
| Weighted cascade F=20 | 0.50 | 8/20 |
| *Follower count alone (top-degree)* | — | 8/20 |

No variant ranks accounts better than raw follower count does. That's a ceiling on what
any optimiser using this simulator can legitimately claim over top-degree.

## What this validates and what it doesn't

It validates the forward model to the degree shown. A floored weighted cascade on the
follower graph reproduces the shallowness of real diffusion out of sample, and two-thirds
of its scale in-window. It does **not** validate any seeding strategy's real-world
effect. Observational data cannot show what would have happened under different seeds,
so strategy comparisons remain simulation results.
