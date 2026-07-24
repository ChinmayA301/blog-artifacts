# Methodology

## Unit of analysis

The unit is one ASER 2024 state or union-territory rural estimate. The published
report contains 27 state/UT pages with sufficient data for both measures.

The panel is not population-weighted. National benchmarks are stored separately
and must not be reconstructed by averaging the state rows.

## Extraction

`extract_aser_2024.py` locates state enrollment pages by two text anchors:
`School enrollment` and `<state> RURAL`. It then extracts:

1. the first five-value row in Table 1, corresponding to ages 6-14; and
2. the third six-value row in Table 4 on the next page, corresponding to
   Standard III.

The enrollment value is `100 - not in school`. The reading value is the fifth
exclusive reading category, `Std II level text`.

The script checks that exactly 27 rows are extracted. Each record retains the
one-based PDF page index and source URL.

## Statistical analysis

The analysis reports:

- minimum, maximum, mean, and coefficient of variation for both measures;
- Pearson product-moment correlation;
- Spearman rank correlation;
- an ordinary least-squares line, reported descriptively;
- a 20,000-shuffle, fixed-seed permutation p-value for the Pearson correlation;
- a 10,000-resample, fixed-seed percentile bootstrap interval.

The permutation test asks whether an association at least as large as the
observed absolute correlation is unusual when state labels are exchangeable.
The bootstrap interval describes sensitivity to resampling these 27 aggregate
rows; it is not a design-based survey confidence interval.

## Interpretation boundary

The study tests whether enrollment distinguishes current reading capability
across states after access is already high. It does not estimate:

- the causal effect of enrollment on reading;
- school quality;
- individual student trajectories;
- urban outcomes;
- learning outside the ASER reading task; or
- whether one state policy caused another state's relative position.

The access-capability wedge is an explanatory device, not a validated composite
index or welfare score.

## Denominator module

The enforcement module performs arithmetic on one official cross-sectional
snapshot. It intentionally avoids statistical claims about political
selectivity because the aggregate source lacks party-coded case-level data,
comparable case vintages, and disposition histories.
