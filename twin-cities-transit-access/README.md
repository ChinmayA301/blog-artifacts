# Twin Cities transit access: a replication exercise

How many jobs can the average Twin Cities worker reach by transit in 30 and 45
minutes, and how does that compare with walking?

This is a **scaled-down replication of the University of Minnesota Accessibility
Observatory's *Access Across America* transit method**, built to learn that method.
It is not original research, and the numbers are not a substitute for AO's
published figures.

- **One-page PDF:** [`outputs/twin_cities_transit_access_onepager.pdf`](outputs/twin_cities_transit_access_onepager.pdf)
- **Interactive map:** <https://chinmaya301.github.io/blog-artifacts/twin-cities-transit-access/> (source: [`index.html`](index.html))
- **Per-block-group results:** [`outputs/bg_accessibility.csv`](outputs/bg_accessibility.csv)

## Provenance tier

**Real public data, method replication.** Every input is a real public dataset
(Metro Transit GTFS, OpenStreetMap, LEHD LODES8, Census TIGER 2020). The method is
AO's, re-implemented at coarser geography. The result is checked against AO's
published numbers, and the differences are listed below rather than tuned away.

## Results

Worker-weighted jobs reachable, averaged over all 120 departure minutes
(LODES 2023: 1,794,825 jobs and 1,613,532 resident workers in the seven counties).

| Minutes | Transit + walk | Walk only | Transit ÷ walk | AO published, Transit 2024 | This run vs AO |
|---|---:|---:|---:|---:|---:|
| 10 | 1,018 | 950 | 1.1× | 409 | +148.9% |
| 20 | 4,056 | 2,459 | 1.6× | 3,737 | +8.5% |
| 30 | 14,785 | 5,282 | 2.8× | 15,664 | -5.6% |
| 40 | 37,468 | 9,501 | 3.9× | 39,282 | -4.6% |
| 45 | 53,475 | 11,988 | 4.5× | – | – |
| 50 | 72,882 | 14,854 | 4.9× | 74,945 | -2.8% |
| 60 | 115,378 | 21,000 | 5.5× | 116,103 | -0.6% |

Rows for 10 and 20 minutes are shown for completeness but are inflated by the block-group
geometry (see below).

**Findings**

1. **Transit multiplies reach, but from a small base.** The average worker can reach 14,785 jobs within 30 minutes by transit and 53,475 within 45, about 2.8x and 4.5x what walking alone reaches. Even at 45 minutes that is 3.0% of the region's 1.79 million jobs.
2. **The gain is concentrated in the two core counties.** Hennepin and Ramsey workers reach 27,861 and 18,342 jobs in 30 minutes by transit. In the five other counties the figure is 1,720, against 1,347 on foot. For 70% of all workers, transit does not even double their 30-minute walking reach.
3. **Departure minute matters, not just location.** For the median block group, 45-minute transit access swings by 56% of its average between the best and worst departure minute in 7-9 AM. That is service frequency showing up in the numbers, which is why AO averages every minute instead of picking one.

**Robustness checks** (all in `outputs/summary.json`)

- *Departure sampling:* averaging only the 24 five-minute departures instead of all 120
  changes the 30-minute figure by -1.0% and the 45-minute figure by
  -0.5%.
- *Destination aggregation:* re-running walking to 19,148 block-level job points
  instead of block-group points changes 30-minute walking access by +3.5% and 45-minute
  by +1.4%. Origins stay at block groups, so this does not test origin aggregation.
- *Outer counties:* Carver and Scott show zero transit gain over walking at 45 minutes. This is not
  a routing bug. SouthWest Transit's trips are in the feed and routed, and 2 Carver block groups
  gain by 60 minutes. Its service is park-and-ride express, though, which a walk-access method
  cannot see. Scott County's MVTA service is not in the feed at all.

## What was replicated, parameter by parameter

| Element | AO method (source) | This run |
|---|---|---|
| Measure | Cumulative opportunities: jobs reachable within *t* minutes (2022 methodology §3.4) | Same, *t* = 0–60 by minute; 30 and 45 are the headline thresholds |
| Time of day | Every departure minute 07:00–08:59, accessibility averaged across them (Transit 2014 methodology §4.4; Transit 2024 report) | Same: all 120 departure minutes routed separately, averaged |
| Routing engine | R5 (2022 methodology §3.3) | r5py 1.1.7, which wraps R5 |
| Walking | 3.6 km/h on pedestrian network for access, egress and transfers (2022 §3.3.4) | Same |
| Transfers | Unlimited (2022 §3.3.4) | Up to 8 rides, which is effectively unlimited at 60 minutes |
| Walk-only trips | Count as transit trips if faster (2022 §3.3.4) | Same (r5 transit mode includes direct walking) |
| Service day | A normal non-holiday Wednesday (2022 §2.5) | Wednesday 7 Oct 2026, from the current Metro Transit feed |
| Jobs | LODES WAC, all jobs, by block | LODES8 WAC 2023, all jobs (S000 / JT00) |
| Person weighting | Workers resident in each block (2022 §3.5) | LODES8 RAC 2023 workers per block group |
| Origins/destinations | Every Census block centroid | 2020 block groups. Origin = worker-weighted mean of block internal points; destination = job-weighted mean |
| Region | Full CBSA (16 counties incl. 2 in Wisconsin) | 7-county Metropolitan Council region |
| Ranking score | Σ (a<sub>t</sub> − a<sub>t−10</sub>) · e<sup>−0.08t</sup> (2022 §3.6) | Computed in `summary.json` as `ao_weighted_score`, but not ranked (one metro only) |

## Where this differs, and why that matters

1. **Block groups, not blocks.** Jobs inside a worker's own block group are counted as
   a few minutes away, which inflates the 10- and 20-minute values. AO's 2014
   methodology makes the same point about zone size versus walking speed. Read the 30-
   and 45-minute results; treat anything below 20 minutes as an artifact.
2. **Seven counties, not the CBSA.** Jobs outside the region are invisible, so access
   near the boundary is understated. Workers in the nine outer CBSA counties, who
   mostly have little transit access, are also excluded. That pushes the regional average up.
3. **MVTA is missing.** The Metro Transit feed bundles Maple Grove, Plymouth, SouthWest
   Transit and U of M routes, but not the Minnesota Valley Transit Authority. Access in
   parts of Dakota and Scott counties is understated.
4. **Schedules, not operations.** Perfect on-time running is assumed. LODES places jobs at
   the employer's reported worksite and cannot see remote or hybrid work.
5. **An open question about AO's own method.** The Transit 2024 report's abstract says
   access is computed from "the 15% fastest travel times", while its body describes
   accessibility averaged over 7–9 AM. This run implements the average.

Items 1 and 2 push the regional average up; items 3 and 4 push it down. The close match
with AO's published figures is partly those effects cancelling out. It is a sanity
check that the pipeline is wired correctly, not a validation of the numbers.

## Reproducing

Needs Python 3.12, a Java 21 JDK for r5py, and `osmium-tool` (`brew install osmium-tool`).
`config.py` looks for a JDK under `~/.local/jdk/jdk-21*`; otherwise set `JAVA_HOME`.
Set `TCA_DATA=/some/path` to keep the ~600 MB of raw and intermediate data outside the repo.
This matters if the checkout is in an iCloud-synced folder, which can evict files under disk pressure.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python 01_download.py         # GTFS, OSM (clipped), TIGER, LODES  ~400 MB
.venv/bin/python 02_prepare_zones.py    # block-group origin/destination points
.venv/bin/python 03_travel_times.py walk
.venv/bin/python 03_travel_times.py blockcheck   # walk to block-level destinations
.venv/bin/python 03_travel_times.py transit      # 120 departures, ~45 s each
.venv/bin/python 04_accessibility.py    # outputs/summary.json, curve.csv, bg_accessibility.csv
.venv/bin/python 05_onepager.py         # the PDF
.venv/bin/python 06_interactive.py      # index.html
```

`03_travel_times.py transit` routes the 5-minute departures first and skips any
departure already on disk, so it can be stopped and resumed. `04_accessibility.py`
reports the 5-minute-sample estimate next to the full 120-minute estimate.

## Files

| File | Role |
|---|---|
| `config.py` | Paths and every routing parameter, each tagged with its AO source or marked as a deviation |
| `01_download.py` … `06_interactive.py` | The pipeline, in order |
| `narrative.py` | Findings and methods text, generated from `summary.json` and shared by the PDF and the web page |
| `interactive_template.html` | Page template; `06_interactive.py` inlines the data |
| `outputs/summary.json` | Every headline number, including the sampling and aggregation checks |
| `outputs/curve.csv` | Worker-weighted jobs reachable at 0–60 minutes, transit and walk |
| `outputs/bg_accessibility.csv` | Per block group: workers, jobs, walk/transit access at 30 and 45 min, best/worst departure minute |

## Sources

- Owen, A., Liu, S., Jain, S., & Lind, E. *Access Across America: 2022 Methodology.* Accessibility Observatory, University of Minnesota (2024).
- Accessibility Observatory. *Access Across America: Transit 2024.* CTS 25-18, University of Minnesota.
- Owen, A., & Levinson, D. *Access Across America: Transit 2014 Methodology.* CTS 14-12, University of Minnesota.
- Fink, C., Klumpenhouwer, W., Saraiva, M., Pereira, R., & Tenkanen, H. *r5py: Rapid Realistic Routing with R5 in Python.* R5 routing engine by Conveyal.
