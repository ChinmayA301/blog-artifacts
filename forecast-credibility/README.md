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
| `forecast_monte_carlo.png` | 200k draws over price/utilisation/adoption multipliers centred at 1.0, correlated vs independent. The headline sits near the median; correlation widens P10–P90 from ~$187–473B to ~$154–563B. |

## Reproducing

```bash
pip install matplotlib numpy
python generate_figures.py
```

## Caveat

The Monte Carlo multipliers are centred at 1.0, so the simulation assumes no
bias in the forecast; the spreads and the correlation structure are
illustrative assumptions, not findings. What the figure isolates is the effect
of correlation between drivers.

**Correction (Nov 2026).** An earlier version set the multipliers' means below
1.0 "to reflect the tendency of AI forecasts to overshoot" and then reported
that ~70% of outcomes fell below the headline. That result was the prior, not
a finding, and the justification conflicted with the post's own evidence that
market-level AI forecasts had undershot.
