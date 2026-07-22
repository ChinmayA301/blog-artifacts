# AI Value Translation Audit

A secondary-data mini-study of a narrow enterprise-AI question:

> When does AI productivity translate into enterprise value?

The study compares nine documented AI deployments and empirical studies using
a five-part audit rubric. It separates local task-productivity gains from
evidence that an AI system improves an operational outcome such as cost, speed,
revenue, quality, risk, satisfaction, or decision performance.

## Evidence scope (read this first)

This is an **author-scored synthesis of secondary sources**, not a causal
meta-analysis or an independent audit of the featured organizations. Reported
outcomes are transcribed from the linked papers and company case studies. The
five audit dimensions and their scores are analytical judgments designed to
make comparisons explicit and reproducible.

The sample is small and purposive. Category averages with one observation
should be treated as descriptions of that case, not population estimates.
Different source designs, outcome definitions, and time horizons also mean the
reported outcome magnitudes should not be compared as if they came from one
controlled experiment.

## Audit method

Each use case receives 0, 1, or 2 points on five dimensions, for a maximum
score of 10.

| Dimension | Scoring question |
|---|---|
| Workflow specificity | Is the AI tied to a repeated operational task? |
| Data readiness | Is it connected to trusted, contextual data? |
| Evaluation loop | Are outputs measured, tested, or A/B validated? |
| Accountability | Is there clear human ownership, review, or escalation? |
| Value translation | Does the gain map to cost, speed, revenue, quality, risk, satisfaction, or decision performance? |

The row-level scores, interpretations, source URLs, and reported outcomes live
in [`ai_value_translation_audit_dataset.csv`](./ai_value_translation_audit_dataset.csv).
The analysis script checks that every total equals the sum of its five
dimensions before producing the summary and charts.

## Use-case results

| Use case | Reported outcome | Score |
|---|---|---:|
| Customer support conversational assistant | +15% average productivity; strongest gains for novice/lower-skilled agents | 9/10 |
| Consulting task assistant | +12.2% more tasks and +25.1% faster; worse performance outside the model frontier | 6/10 |
| Coding assistant controlled task | Developers completed a bounded JavaScript task 55.8% faster | 6/10 |
| Customer support AI agents at bank scale | +37 percentage points AI transactional NPS; +29 percentage points self-service rate | 10/10 |
| Online retail GenAI workflow experiments | 0% to 16.3% sales lift depending on workflow; about $5 annual incremental value per consumer in positive cases | 9/10 |
| Klarna customer service AI assistant | 2.3M conversations in the first month; 25% fewer repeat inquiries; projected $40M profit improvement | 7/10 |
| Chime AI-powered support and disputes | Satisfaction +80%; dispute resolution time down more than 50%; fraud losses down 29% | 9/10 |
| Morgan Stanley advisor knowledge assistant | 98%+ advisor-team adoption; document access increased from 20% to 80% | 9/10 |
| Alibaba after-sales support assistant | Faster service and higher subjective quality; no significant objective retrial effect; declines among some top performers | 8/10 |

## Category summary

| Category | N | Average score | Range |
|---|---:|---:|---:|
| Applied workflow AI | 4 | 9.25/10 | 9-10 |
| Knowledge integration AI | 1 | 9.00/10 | 9-9 |
| Mixed applied workflow AI | 1 | 8.00/10 | 8-8 |
| High-output automation | 1 | 7.00/10 | 7-7 |
| Local productivity AI | 2 | 6.00/10 | 6-6 |

## Figures produced

The analysis script generates two local outputs. PNG files are ignored by this
repository because publication figures live with the blog assets.

| File | What it shows |
|---|---|
| `audit_scores_by_use_case.png` | The nine audited use cases ranked by total score |
| `category_average_scores.png` | Average audit score for each analytical category |

## Findings

1. **Productivity is not the same as enterprise transformation.** The task
   studies show real gains under specific conditions, but a faster local task
   does not by itself establish organization-level value.
2. **The strongest cases are bounded workflow systems.** The highest-scoring
   examples combine a defined task, contextual data, measurable outcomes, and
   a path from model output to operational performance.
3. **Output without ownership is a risk.** The Klarna and Alibaba cases show
   why automation volume or subjective quality can coexist with accountability
   concerns, mixed objective effects, or performance declines in some groups.
4. **Readiness is the translation layer.** Organizations need evaluation,
   ownership, integration, and monitoring to connect model capability to cost,
   quality, speed, risk, revenue, or customer outcomes.

## Connection to Aegis

This study motivates the Aegis AI Readiness Audit, which evaluates proposed AI
use cases across workflow fit, data readiness, evaluation design, human
accountability, risk and governance exposure, ROI translation, integration
complexity, and lifecycle ownership.

The governing thesis is:

> AI value is not determined by model capability alone. It depends on whether
> the organization can translate that capability into governed workflow
> performance.

## Repository contents

| File | Purpose |
|---|---|
| `README.md` | Study question, method, results, caveats, and sources |
| `ai_value_translation_audit_dataset.csv` | Nine scored use cases with source links and interpretations |
| `ai_value_translation_audit_analysis.py` | Score validation, category aggregation, and chart generation |
| `category_summary.csv` | Generated category-level counts and score statistics |
| `audit_scores_by_use_case.png` | Generated local use-case score chart (gitignored) |
| `category_average_scores.png` | Generated local category-average chart (gitignored) |

## Reproducing

From this directory:

```bash
pip install -r ../requirements.txt
python ai_value_translation_audit_analysis.py
```

The script reads the row-level dataset, validates each audit total, rewrites
`category_summary.csv`, and regenerates both gitignored PNG charts in place.

## Sources

- Brynjolfsson, Li, and Raymond, [*Generative AI at Work*](https://arxiv.org/abs/2304.11771)
- Dell'Acqua et al., [*Navigating the Jagged Technological Frontier*](https://www.hbs.edu/faculty/Pages/item.aspx?num=64700)
- Peng et al., [*The Impact of AI on Developer Productivity*](https://arxiv.org/abs/2302.06590)
- Gupta et al., [*Building Customer Support AI Agents at 100M-User Scale*](https://arxiv.org/abs/2606.08867)
- Fang et al., [*Generative AI and Firm Productivity*](https://arxiv.org/abs/2510.12049)
- OpenAI, [Klarna case study](https://openai.com/index/klarna/)
- Chime, [How Chime is using AI to redefine member experience](https://www.chime.com/newsroom/how-chime-is-using-ai-to-redefine-member-experience/)
- OpenAI, [Morgan Stanley case study](https://openai.com/index/morgan-stanley/)
- Ni et al., [*Generative AI in Action*](https://arxiv.org/abs/2603.29888)
