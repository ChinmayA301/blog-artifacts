# Forecast Credibility

Code behind the figures in
[Where Do the AI Numbers Actually Come From?](https://app.chinmayarora.com/blog/ai-forecast-numbers/)

The post stress-tests Goldman Sachs's projection that SpaceX AI revenue rises
from **$3.2B (2025) to $322B (2030)** — a ~100× jump, ~151% CAGR — against the
company's own S-1 filings and FT reporting.

## Figures produced

`generate_figures.py` writes three PNGs to `./figures/`:

| File | What it shows |
|---|---|
| `forecast_cagr_waterfall.png` | Decomposes the 100× build into yearly increments; the bulk lands in 2029–30. |
| `forecast_path_comparison.png` | Goldman's front-then-back-loaded path vs. a constant-CAGR path — where the forecast hides its mass. |
| `forecast_monte_carlo.png` | 200k correlated draws over price/utilisation/adoption multipliers; ~70% of outcomes fall below the $322B headline. |

## Reproducing

```bash
pip install matplotlib numpy
python generate_figures.py
```

## Caveat

The Monte Carlo priors (means set below 1.0, the correlation structure) are
grounded in the documented historical tendency of AI forecasts to overshoot,
but they are assumptions, not findings — the figure caption states this. To
harden the result, recalibrate the overshoot multipliers to a specific
historical forecast cohort.
