# Data provenance

`generate_figures.py` downloads public StatsBomb open-data JSON from:

https://github.com/statsbomb/open-data

The script caches raw JSON under `../.cache/statsbomb-open-data/` and writes a
derived analysis table to `forward_pressing_context_dataset.csv`. The derived
CSV is intentionally ignored by git because it is a generated artifact.

## Scope

This is an exploratory public-data MVP, not a full tracking-data audit. The
available event coverage is uneven: some competitions are full seasons, some
are partial player/team samples, and Champions League coverage is mostly single
finals. The study therefore tests the article's measurement claim under public
data constraints rather than claiming a definitive all-competition estimate.

## Variables

The derived table is aggregated at player-team-competition-season level for a
selected set of elite forwards. It includes:

- player pressure events per 90
- counterpresses per 90
- goals, non-penalty xG, shots, box touches, and progressive receptions per 90
- team points per match while the player appears in the sample
- team xG difference, pass-share possession proxy, and a PPDA-style pressure
  context proxy from event data

Tracking-only features such as line height, cover shadow, velocity, and
time-to-intercept are not available in this public feed, which is part of the
point of the study.
