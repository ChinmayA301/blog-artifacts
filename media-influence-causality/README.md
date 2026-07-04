# Media Influence — Correlation, Lag, and the Causation Problem

Code behind the figures in the American-media-and-global-power post
([app.chinmayarora.com/blog](https://app.chinmayarora.com/blog/)).

Does media export *drive* global influence, or just move with it? This artifact
does the honest version of that question: it measures the lead/lag relationship
with a cross-correlation function, then uses a causal diagram to make explicit
why a lagged correlation — however clean — does not settle the causal claim.

## Data caveat (read this first)

**The data here is synthetic**, constructed so that a "media export" series
genuinely leads an "influence index" by a known lag, with an added confounder.
The *analysis* (cross-correlation with significance band, plus the causal DAG)
is the real artifact. Captions state this.

To promote this to a real finding, assemble public series (e.g. media/cultural
export volumes and a soft-power or influence index), run the same CCF, and —
crucially — bring an **instrument or natural experiment** for the causal step.
The DAG shows exactly why more correlation won't substitute for that.

## Figures produced

`generate_figures.py` writes to `./figures/`:

| File | What it shows |
|---|---|
| `media_influence_ccf_dag.png` | Left: cross-correlation with a clear peak at a positive lag (media leads), inside a 95% no-correlation band. Right: a causal DAG laying out the three competing stories — direct effect, confounding, and reverse causation. |

## The point of the DAG

The cross-correlation is *consistent with* media leading influence, but it is
equally consistent with a shared confounder (economic/soft power) driving both,
or with reverse causation. The diagram exists to keep the empirical result from
being oversold — the honest contribution is framing the identification problem,
not claiming to have solved it.

## Reproducing

```bash
pip install matplotlib numpy scikit-learn
python generate_figures.py
```
