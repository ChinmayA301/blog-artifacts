# Successful India™ — Audit Report & Study Runs

*A verification pass over every claim in the blog + dashboard, plus two original studies (A and C) actually run on public data. Written to the same standard as the pieces themselves: findings that cut against the thesis are reported as prominently as findings that support it.*

---

## Part 1 — The Audit

**43 claims tracked. 31 Tier-1 (primary/independent). 12 Tier-2 (secondary/proprietary/private). 2 material corrections found. 22 flagged for action before publishing.**

The full machine-readable tracker is in `successful-india-source-tracker.csv` — one row per claim, with provenance tier, verification status, an audit note, and the specific action needed. Here is what matters most.

### The two material corrections (these are real errors in the current pieces)

**Correction 1 — the headline education number is mislabelled.** This is the big one. The primary ASER 2024 PDF (asercentre.org) states explicitly, for each figure, that **"23.4%" refers to government-school Std III students specifically — not all children.** The all-India figure across *all* schools (government + private) for Std III reading recovered to roughly **27.1%** (Ideas for India, reading the same ASER data). My blog and dashboard present 23.4% as the all-children national figure. That's wrong.

- The arithmetic figure (33.7% subtraction) *is* genuinely all-India — the PDF confirms it.
- But the Class-5 reading (44.8%) and Class-5 division (30.7%) are again **government-school** figures.
- So the pieces mix all-India and government-school numbers without labelling which is which, and present at least one government-school number as if it were national.

**Fix:** either (a) label every figure with its school-type, or (b) switch the headline to the ~27.1% all-India number. Option (b) is cleaner but slightly less dramatic — which is exactly why the honest choice is to state both and let the government-school number stand as the sharper sub-finding. The "3 in 4 can't read" line should be tied explicitly to government schools, or softened to the ~73% all-India inverse.

**Correction 2 — the "95% opposition" figure is not government data.** The primary source (Ministry of Finance reply, via Outlook's verbatim report) is unambiguous: the government **"does not keep any breakdown of these cases by political party, state, or whether the individuals involved belonged to the ruling or opposition benches."** So the 193-cases/2-convictions figure *is* government data, but the "95% target the opposition" figure is **not** — it comes from a 2022 *Indian Express* investigation and a 14-party Supreme Court plea. My pieces present them side by side in a way that implies both are the government's own admission. They must be separated: one is the state's data, the other is journalism/litigation.

The refusal to publish the party-wise breakdown is itself worth naming — it's a finding, not just a gap.

### The tiering, honestly

**Tier 1 — verified, load-bearing, defensible** (the spine of the whole thesis survives the audit):
- Electoral bonds unconstitutional ruling — a public Supreme Court judgment.
- ED 193/2 conviction data — the government's own Parliament reply, confirmed verbatim.
- ASER learning levels — independent survey, primary PDF read directly (with the labelling correction above).
- IIT ability-graded migration (62% of top-100) — peer-reviewed, *Journal of Development Economics*.
- NFAP unicorn-founder counts, IFP AI-founder counts, GERD 0.64%, V-Dem/Freedom House/RSF classifications — all independent, multi-source confirmed.

**Tier 2 — real but needs attribution or origin-tracing before terminal citation:**
- Das–Drèze wage figures → trace to their original paper, not Countercurrents.
- SBI profit-to-GDP → attribute as "per SBI Research," not fact.
- "80% engineers unemployable" → Aspiring Minds, a commercial firm; attribute.
- Henley millionaire figures → proprietary, methodology contested; attribute and keep the both-sides caveat.
- Researcher-density (262/million) → trace to UNESCO UIS.
- RDI-fund 15%-utilisation → trace to the Budget/PIB document.
- Stanford AI Vibrancy exact scores, IndiaAI GPU counts, BHASHINI inference volume → trace to primary before quantifying.

**Honest meta-point:** "every number audited" means I re-fetched the primary source where it was fetchable this session (ASER PDF, the ED Parliament reply) and confirmed multi-source consensus for the rest. Several Tier-2 items live in proprietary or paywalled reports (Henley, Aspiring Minds, some Economic Survey annexures) that I could not open directly — for those, "audited" honestly means "traced to the closest verifiable public record," and the tracker says so per-row rather than pretending otherwise. The load-bearing claims are Tier-1 and verified. The texture includes Tier-2 that should be traced before publication.

---

## Part 2 — The Studies (actually run)

Both studies were executed on real public data. The scripts are in the repo (`study_A_metric_substitution.py`, `study_C_enforcement.py`) and reproducible. Crucially, **both produced at least one result that cut against the thesis, and both writeups report it prominently** — that's the point of running them rather than asserting them.

### Study A — "Metric Substitution" (the mission-statement thesis, made testable)

**Design.** The thesis is: *India optimises measurable proxies while the underlying capability decays or lags.* The novel move — not in any single cited study — is to put a **proxy** metric and its **capability** metric in the same frame and measure the *wedge* between them. Education is the proof-of-concept domain because it has clean public state-level data: proxy = enrolment ("are children in the building?"), capability = ASER foundational learning ("can they read?"). Run on real ASER 2018→2024 state figures.

**Results:**

1. **The saturation result (strong, supports thesis).** Across states, enrolment has a coefficient of variation of **0.009** — it's ~98% everywhere, essentially flat. Learning has a CV of **0.430** — ~45× more variance. The metric we celebrate has stopped discriminating between good and bad systems; the metric we ignore carries almost all the signal. Optimising a saturated proxy is, definitionally, metric substitution.

2. **The wedge (strong, supports thesis).** The average state puts ~98% of children in school and ~35% over the foundational-reading bar — a **~63-point wedge** between "looks successful" and "is successful."

3. **The correlation result (honest — partly *cuts against* the thesis).** Enrolment vs learning across states came out at **r = 0.64** — *moderate, not near-zero.* My first draft of the script narrated this as "decoupled / barely predicts," which the number does **not** support. Corrected: enrolment explains ~42% of learning variance (r² = 0.42), so it's a *weak* predictor, not a useless one. Partial decoupling, not total. The saturation leg is the strong evidence; the correlation leg is real but softer than a motivated reading would want — and reporting it straight is the whole point.

4. **The honest counter-test (supports fairness).** Capability *is* improving — 8/8 sampled states rose 2018→2024, a mean +9.2 points. The genuine ASER-2024 recovery is real. So the indictment is the *level* (~35% over the bar even after the best gains in 20 years) and the *saturation of the proxy*, **not** stagnation.

**Verdict:** the metric-substitution thesis is measurable and supported in education — but with one leg (saturation) strong and one leg (correlation) modest, and with an honest credit to real improvement. **Generalisation** (the actual paper): replicate the wedge across domains — GDP-growth vs real-wage-growth, R&D-allocation vs R&D-utilisation. The thesis predicts a positive, widening wedge in each. **The real contribution is building the data** — a state-year panel joining UDISE+ enrolment, ASER learning, PLFS wages, MoSPI GSDP, and DST R&D-utilisation. That table does not appear to exist publicly. Building it *is* the study.

### Study C — "Selective Enforcement" (the accountability limb)

**Design.** If enforcement agencies are instruments of power rather than law, then (P1) their activity should be *decoupled from conviction outcomes*, and (P2) their activity should be *asymmetrically distributed against the opposition beyond base rates*. Tested on discrete public data only: the ED Parliament reply (verified primary), electoral-bond disclosures (court-ordered), and — for P2, honestly flagged as weaker — the *Indian Express* / SC-plea figures.

**Results:**

1. **P1 — decoupling (strongly supported, on verified government data).** Politician cases convict at **1.04%** (2 of 193) versus the overall PMLA rate of **4.61%** (42 of 911) — politician cases convict at just **22% of an already-dismal baseline.** A binomial test: if politician cases converted at the overall PMLA rate, you'd expect ~8.9 convictions; observed = 2; **P(≤2) = 0.006.** So the shortfall is statistically significant, not noise. But the real story is the *baseline itself* — a coercive, bail-denying process with a ~1% conviction rate is, functionally, a process whose output is the process (the raids, the arrests, the years on bail), not the verdict.

2. **P2 — asymmetry (supported, but on weaker journalistic data).** Opposition share of politician probes rose from ~54% (pre-2014, roughly a neutral base rate given incumbency) to ~95% (post-2014) — an **odds ratio of ~16×.** But this rests on the *Indian Express* count, **not** government data, because the government refuses to publish the party-wise breakdown. The refusal is itself a finding.

3. **P3 — co-movement (suggestive).** Two *independently sourced* public datasets align: the party receiving ~48% of all opaque electoral-bond money is the same party whose rivals absorb ~95% of coercive enforcement. Resources flow toward power; coercion flows away from it. That co-movement is the quantitative signature of the mission thesis at the accountability layer.

**Verdict:** P1 is a clean, publishable result built entirely on the state's own numbers. P2/P3 are directionally strong but need case-level microdata (filing date, party-at-filing, outcome) to move from "suggestive" to "demonstrated." **What would falsify it:** case-level data showing opposition probe-share tracking their share of political actors, or politician-case conviction rates comparable to non-political PMLA cases. The government holds that data and has not released it.

---

## Bottom line

The audit found two real errors and a dozen attribution gaps — none of which collapse the thesis, all of which are fixable, and finding them is exactly why the audit was worth running. The two studies turn the essay's rhetoric into two falsifiable, partly-successful analyses: the metric-substitution wedge is real and measurable (with one honest soft leg), and the enforcement decoupling is statistically significant on the government's own data. Both are publishable as short research artifacts now; both point to a single, larger, genuinely novel contribution — **assembling the cross-domain proxy-vs-capability panel that nobody currently maintains.**
