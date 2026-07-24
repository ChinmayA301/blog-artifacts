# Metric Substitution Audit

A reproducible pilot study of a narrow question:

> When a visible proxy is nearly universal, does it still tell us much about
> the underlying capability it is supposed to represent?

The empirical module uses the full set of 27 state and union-territory pages in
the ASER 2024 rural report. It compares school enrollment among children aged
6-14 with the percentage of Standard III children who can read a Standard II
text. A second, separate denominator lab uses official Enforcement Directorate
aggregates to show why the same numerator can support very different
percentages without supporting the same claim.

## Bottom line

- Rural enrollment is compressed into a narrow **95.9%-99.9%** state range.
- Grade III reading spans **6.2%-50.6%** across the same 27 states/UTs.
- Enrollment and reading have a weak cross-sectional association:
  **Pearson r = 0.11**, permutation **p = 0.59**, with a bootstrap 95% interval
  of **-0.23 to 0.44**.
- Reading's coefficient of variation is about **37 times** enrollment's.
- This supports a limited metric-substitution claim: once access is saturated,
  enrollment alone has little discriminatory power for current learning
  capability.

This is not evidence that enrollment is unimportant, that learning is caused
by enrollment policy, or that the calculated wedge is a causal treatment
effect.

## A material source correction

ASER's **23.4%** headline is the 2024 figure for government-school Standard III
students who can read a Standard II text. It is not the all-children national
figure. The corresponding ASER table reports **27.0% for all children** and
**27.1% for the government-plus-private weighted estimate**.

The analysis and dashboard use the all-children state-table measure.

## Study design

### Hypothesis

If enrollment has become a saturated access proxy, cross-state variation in
enrollment should be much smaller than variation in foundational reading, and
enrollment should weakly distinguish high- from low-learning states.

### Measures

| Construct | Measure | ASER table |
|---|---|---|
| Access proxy | Percentage of children age 6-14 enrolled in school | State Table 1 |
| Capability outcome | Percentage of Std III children reading a Std II text | State Table 4 |
| Descriptive wedge | Enrollment minus reading, in percentage points | Derived |

### Tests

1. Compare ranges and coefficients of variation.
2. Estimate Pearson and Spearman cross-state associations.
3. Use a fixed-seed permutation test for the Pearson correlation.
4. Bootstrap the Pearson correlation to show sampling uncertainty.

The 27 rows are geographic aggregates, not independent randomized units.
Results are descriptive and cross-sectional.

## Enforcement denominator lab

Official ED statistics as of 31 March 2026 report 8,851 ECIRs recorded, 2,396
prosecution complaints, 60 completed trials, 56 conviction outcomes, and four
acquittals.

Using 56 as the numerator produces:

| Denominator | Percentage | What it describes |
|---|---:|---|
| 60 completed trials | 93.33% | Share of concluded trial cases ending in conviction |
| 2,396 prosecution complaints | 2.34% | Conviction outcomes relative to complaints filed |
| 8,851 ECIRs recorded | 0.63% | Conviction outcomes relative to recorded investigations |

These are not competing estimates of one quantity. They answer different
pipeline questions and mix cases of different ages. Aggregate ED data also do
not contain party-coded case records, so they cannot by themselves test
selective partisan enforcement. That study remains feasible only after
assembling a verified case-level panel with party, timing, stage, disposition,
and comparable exposure fields.

## Repository contents

Everything associated with the study lives inside this folder.

| Path | Purpose |
|---|---|
| `README.md` | Research question, findings, caveats, and reproduction |
| `extract_aser_2024.py` | Deterministic extractor for ASER state tables |
| `analyze_metric_substitution.py` | Statistical analysis and figure generation |
| `data/aser_2024_state_proxy_capability.csv` | Extracted 27-state panel |
| `data/national_benchmark.csv` | National ASER benchmark and scope labels |
| `data/ed_denominator_snapshot.csv` | Official ED aggregate snapshot |
| `data/source_audit.csv` | Claim-level verification and correction tracker |
| `state_wedge_summary.csv` | States sorted by the descriptive wedge |
| `results.json` | Machine-readable results used by the dashboard |
| `docs/methodology.md` | Extraction, statistical, and interpretation details |
| `docs/source-notes.md` | Source provenance and known limits |
| `figures/` | Generated charts committed with this self-contained study package |

## Reproducing

Install the repository requirements from this directory:

```bash
pip install -r ../requirements.txt
```

Download the primary ASER report, then extract and analyze:

```bash
curl -L \
  "https://asercentre.org/wp-content/uploads/2022/12/ASER_2024_Final-Report_13_2_24-1.pdf" \
  -o ASER_2024_Final-Report.pdf

python extract_aser_2024.py ASER_2024_Final-Report.pdf
python analyze_metric_substitution.py
```

Expected source SHA-256 for the report used in this run:

```text
402808c3bd282686333e41b700dcaa8942bb21094b4daa59672606f2047d4195
```

The analysis writes `results.json`, `state_wedge_summary.csv`, and the two
figures committed in this folder.

## Primary sources

- [ASER 2024 report and state pages](https://asercentre.org/aser-2024/)
- [ASER 2024 national findings](https://asercentre.org/wp-content/uploads/2022/12/ASER-2024-National-findings.pdf)
- [ASER 2024 all-India tables](https://asercentre.org/wp-content/uploads/2022/12/India.pdf)
- [Directorate of Enforcement statistics](https://enforcementdirectorate.gov.in/performance/statistics/)
- [FATF/APG/EAG 2024 mutual evaluation of India](https://www.enforcementdirectorate.gov.in/media/mer/7a11bcc5-23ec-4a4e-ab93-2e44b378fb4b_India-MER-2024.pdf.coredownload.inline.pdf)
- [Election Commission electoral-bond disclosure](https://www.eci.gov.in/disclosure-of-electoral-bonds)
