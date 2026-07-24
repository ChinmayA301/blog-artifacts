# Successful India™

> A country winning at everything except the things that compound.

This folder is the complete research package for
[Successful India™](https://app.chinmayarora.com/blog/metric-substitution-audit/).
The data essay is the primary study. The audit, scripts, source tracker,
full-panel ASER analysis, and interactive dashboard are companion evidence
that make its claims inspectable and falsifiable.

The framing is metric substitution: the systematic optimization of a visible
proxy while the capability that proxy is meant to represent lags. The essay
follows that pattern across enrolment and learning, GDP and wages, credentials
and employability, research allocation and research output, retained talent,
and institutional accountability.

## Package contents

| Path | Role |
|---|---|
| `successful-india-blog.md` | Canonical essay source, preserved in the author's original voice |
| `successful-india-dashboard.html` | Original standalone dashboard supplied with the essay |
| `successful-india-source-tracker.csv` | Claim-level tracker covering 43 claims, provenance tiers, corrections, and follow-up actions |
| `audit-and-studies-memo.md` | Original audit and study memo |
| `study_A_metric_substitution.py` | Original eight-row exploratory education analysis, retained for provenance |
| `study_C_enforcement.py` | Original exploratory enforcement analysis, retained for provenance |
| `companion-audit/` | Re-extracted 27-state ASER panel, denominator audit, methods, CSVs, results, and checked figures |

Everything associated with the essay and its validation lives inside this
single folder.

## The data-science result used in the essay

The companion analysis replaces the hand-entered eight-row proof of concept
with the full set of 27 rural state and union-territory panels published in
ASER 2024.

- Enrolment spans **95.9%–99.9%**.
- Class 3 reading spans **6.2%–50.6%**.
- Reading's coefficient of variation is about **37 times** enrolment's.
- Pearson **r = 0.11**, fixed-seed permutation **p = 0.59**.
- Bootstrap 95% interval: **−0.23 to 0.44**.
- Spearman **ρ = −0.003**; simple-model **R² = 0.012**.

This supports a bounded conclusion: once access is saturated, enrolment is a
weak cross-state discriminator of current foundational reading. It does not
estimate a causal effect or rank school policy.

## Why there are two generations of Study A

`study_A_metric_substitution.py` is the original exploratory script and remains
unchanged so the research history is auditable. Its small hand-entered sample
reported a 45× variation ratio and `r = 0.64`.

`companion-audit/analyze_metric_substitution.py` is the publication-grade
companion run. It uses the complete extracted ASER panel and produces the 37×,
`r = 0.11` result reported in the published essay. The larger, source-retaining
panel supersedes the exploratory estimate for the education section.

## Enforcement interpretation boundary

The original Study C script is also retained unchanged. The companion audit
does not reproduce a partisan-selectivity result because the official
aggregate source has no party-coded case records or case-level histories.

It instead demonstrates denominator discipline using the official snapshot:
56 conviction outcomes are 93.33% of 60 completed trials, 2.34% of 2,396
prosecution complaints, and 0.63% of 8,851 ECIRs. These ratios answer different
pipeline questions. A defensible selectivity study still needs verified
case-level timelines, party status at each event, stages, outcomes, exposure
groups, and a pre-specified estimand.

## Reproduce the full-panel companion

From this folder:

```bash
pip install -r ../requirements.txt
cd companion-audit
curl -L \
  "https://asercentre.org/wp-content/uploads/2022/12/ASER_2024_Final-Report_13_2_24-1.pdf" \
  -o ASER_2024_Final-Report.pdf
python extract_aser_2024.py ASER_2024_Final-Report.pdf
python analyze_metric_substitution.py
```

Expected SHA-256 for the ASER report used in the published run:

```text
402808c3bd282686333e41b700dcaa8942bb21094b4daa59672606f2047d4195
```

## Primary evidence

- [ASER 2024 report and state pages](https://asercentre.org/aser-2024/)
- [ASER 2024 national findings](https://asercentre.org/wp-content/uploads/2022/12/ASER-2024-National-findings.pdf)
- [ASER 2024 all-India tables](https://asercentre.org/wp-content/uploads/2022/12/India.pdf)
- [Directorate of Enforcement statistics](https://enforcementdirectorate.gov.in/performance/statistics/)
- [FATF/APG/EAG mutual evaluation of India](https://www.enforcementdirectorate.gov.in/media/mer/7a11bcc5-23ec-4a4e-ab93-2e44b378fb4b_India-MER-2024.pdf.coredownload.inline.pdf)
- [Election Commission electoral-bond disclosure](https://www.eci.gov.in/disclosure-of-electoral-bonds)
