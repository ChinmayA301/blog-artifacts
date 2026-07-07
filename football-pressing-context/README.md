# Football Pressing Context

Code behind the "pressing is not a morality stat" study for
[app.chinmayarora.com/blog](https://app.chinmayarora.com/blog/).

The question is not whether pressing matters. It does. The study asks a narrower
data-science question: does raw forward pressure volume explain team outcomes,
or does the interpretation change once attacking tradeoff and team style enter
the model?

## Data caveat (read this first)

This is a **real public-data exploratory study** built from StatsBomb open event
data. It is not a definitive Opta/Wyscout/SkillCorner-style season audit.

The public coverage is uneven. This script uses full La Liga 2015/16 coverage,
partial Bundesliga/Ligue 1 club samples, recent World Cups and Euros, and a
small set of Champions League finals. There is also no player tracking, so
spatial coordination, velocity, distance, cover shadow, defensive-line height,
and time-to-intercept cannot be modeled directly.

That limitation is central to the article's claim: a pressure count is an event
label; pressing value is spatial, temporal, collective, and role-dependent.

## Figures produced

`generate_figures.py` writes to `./figures/`:

| File | What it shows |
|---|---|
| `pressing_naive_relationship.png` | Raw forward pressures per 90 vs team points per match, with role groups and scoring output. |
| `pressing_attacking_tradeoff.png` | Pressures per 90 vs goals plus non-penalty xG per 90, colored by team outcome tier. |
| `team_style_interaction.png` | Player pressure volume against a PPDA-style team context proxy. |
| `ronaldo_timeline.png` | Ronaldo's public-data samples across competitions/seasons, showing pressure volume alongside attacking output. |
| `context_ladder.png` | The measurement ladder from raw pressure count to role/system/outcome-adjusted pressure value. |

The script also writes `model_summaries.txt` with four OLS models:

1. Naive pressure-only model
2. Attacking-tradeoff model
3. Team-context model
4. Pressure x team-style interaction model

## Reproducing

```bash
pip install -r ../requirements.txt
python generate_figures.py
```

The first run downloads and caches StatsBomb open-data JSON. Later runs reuse
the local cache.

## Expected interpretation

The artifact is designed to test this article line:

> Low pressing is a cost. The question is whether the system can afford it and
> whether the player pays it back elsewhere.

If the pressure-only model is weak or changes materially after adding attacking
and team-context variables, the correct conclusion is not "pressing does not
matter." It is that raw pressure volume is incomplete as a player-value claim.

## Latest local run

Final verification run: 2026-07-07.

- Derived sample: 43 player-team-competition-season rows
- Players covered: 11 selected forwards
- Naive pressure-only model: R2 = 0.017, pressure coefficient = -0.0193, p = 0.403
- Attacking-tradeoff model: R2 = 0.161, pressure coefficient = -0.0009, p = 0.971
- Team-context model: R2 = 0.364, pressure coefficient = 0.0186, p = 0.523
- Interaction model: R2 = 0.365, pressure coefficient = -0.0106, p = 0.916

In this public-data MVP, raw forward pressure volume is a weak standalone
predictor of team points. Adding attacking output and team-context proxies
improves model fit, while the pressure coefficient itself remains unstable and
not statistically meaningful. That supports the article's disciplined claim:
raw pressure count is a real input, but it is not a complete explanation of
player value or team outcomes.
