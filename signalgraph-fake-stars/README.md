# SignalGraph — Detecting Inorganic GitHub Stars

Code behind the figures in the SignalGraph post
([app.chinmayarora.com/blog](https://app.chinmayarora.com/blog/)).

A GitHub star is a vanity metric, and vanity metrics get gamed. This artifact
builds a detector that separates organic star growth from purchased bursts, and
shows how discounting the inorganic stars reshuffles a credibility ranking.

## Data caveat (read this first)

GitHub's per-stargazer timestamp API requires authentication and meaningful
request volume, which was not available in the environment where these figures
were produced. **The data here is synthetic**, with deliberately planted
inorganic bursts. The *detection method* is the real contribution; every figure
caption states that the data is illustrative.

To promote this to a real finding, pull actual `starred_at` timestamps via the
authenticated GitHub API (`Accept: application/vnd.github.star+json`) and run
the same detector on real repos.

## Figures produced

`generate_figures.py` writes to `./figures/`:

| File | What it shows |
|---|---|
| `signalgraph_velocity.png` | Robust median/MAD z-score on daily star gains, with burst-clustering; catches two planted spikes, zero false positives on organic growth. |
| `signalgraph_ranking.png` | Raw star rank vs. credibility-adjusted rank; repos with >20% burst-sourced stars drop sharply. |

## Method

The detector computes a **robust z-score** (median/MAD, resistant to the spikes
themselves) on day-over-day star gains, then applies **burst-clustering** —
only contiguous runs of anomalous days are flagged, which removes isolated
false positives. Stars attributable to flagged bursts are then discounted to
produce a credibility-weighted count.

## Reproducing

```bash
pip install matplotlib numpy
python generate_figures.py
```
