#!/usr/bin/env python3
"""
STUDY C - "Selective Enforcement": a quantitative test using only public, discrete data.

THESIS (mission statement, accountability limb):
  If enforcement agencies are being used as instruments of power rather than
  of law, then (a) their activity should be decoupled from their conviction
  outcomes, and (b) their activity should be asymmetrically distributed against
  the political opposition, beyond what base rates would predict.

This is deliberately modest: it uses ONLY figures that are on the public record
(government Parliament replies + court-disclosed data + one journalistic count),
and it tests falsifiable propositions rather than asserting intent.

HONEST LIMITS (stated up front, not buried):
  - The government explicitly does NOT publish a party-wise breakdown of ED cases.
    The 95% opposition figure is from the Indian Express (2022) + a 14-party SC
    plea, NOT government data. So Proposition 2 is tested against JOURNALISTIC
    data and is correspondingly weaker than Proposition 1.
  - n is tiny and these are aggregates, not case-level microdata. This is a
    "descriptive + base-rate" analysis, not causal inference. A real paper would
    need the case-level dataset (filing date, party at filing, outcome).
"""
import math

print("="*70)
print("STUDY C — SELECTIVE ENFORCEMENT: A BASE-RATE TEST ON PUBLIC DATA")
print("="*70)

# ---------------------------------------------------------------
# PROPOSITION 1: Activity is decoupled from outcomes (conviction).
# Source: Ministry of Finance, Rajya Sabha reply, March 2025 (VERIFIED to primary).
# ---------------------------------------------------------------
print("\n--- Proposition 1: Is ED activity decoupled from convictions? ---")
ed_pol_cases = 193          # cases vs politicians, 10 yrs
ed_pol_convictions = 2      # convictions
pmla_total = 911            # all PMLA cases Jan2019-Oct2024
pmla_convictions = 42       # convictions in that window

pol_conv_rate = ed_pol_convictions/ed_pol_cases
pmla_conv_rate = pmla_convictions/pmla_total
print(f"  ED cases vs politicians: {ed_pol_cases} -> convictions: {ed_pol_convictions}")
print(f"  Conviction rate (politician cases): {pol_conv_rate*100:.2f}%")
print(f"  Conviction rate (all PMLA, 2019-24): {pmla_conv_rate*100:.2f}%")

# A functioning prosecutorial system, even a slow one, would show politician-case
# conviction rates NOT wildly below the (already low) overall rate. Ratio test:
ratio = pol_conv_rate / pmla_conv_rate if pmla_conv_rate else float('inf')
print(f"  Politician-case conviction rate is {ratio:.2f}x the overall PMLA rate.")
print(f"  Interpretation: politician cases convict at ~{ratio*100:.0f}% of the")
print(f"    (already dismal) baseline. The process, not the verdict, is the output.")

# Simple binomial sanity check: if the 'true' success prob were the overall PMLA
# rate (4.6%), how surprising is 2/193?
p = pmla_conv_rate
n = ed_pol_cases
expected = n*p
# probability of <=2 successes under Binomial(193, 0.046)
def binom_pmf(k,n,p):
    return math.comb(n,k)*(p**k)*((1-p)**(n-k))
cdf2 = sum(binom_pmf(k,n,p) for k in range(0,3))
print(f"  If politician cases converted at the overall PMLA rate ({p*100:.1f}%),")
print(f"    expected convictions = {expected:.1f}. Observed = 2.")
print(f"    P(<=2 | Binom(193, {p:.3f})) = {cdf2:.3f}")
print(f"    -> {'Observed is below but not statistically shocking vs the low baseline.' if cdf2>0.05 else 'Observed is significantly below even the low baseline.'}")
print("    NOTE: the striking fact is the BASELINE itself, not the deviation:")
print("    a >95% failure rate on a coercive, bail-denying process is the story.")

# ---------------------------------------------------------------
# PROPOSITION 2: Activity is asymmetric against opposition beyond base rates.
# Source: Indian Express (2022) + 14-party SC plea. NOT govt data.
# ---------------------------------------------------------------
print("\n--- Proposition 2: Is targeting asymmetric vs the opposition? ---")
print("  [Data = journalistic/litigant, NOT govt. Weaker tier. Stated honestly.]")

opp_share_pre2014 = 0.54    # CBI: opposition share of politician probes pre-2014
opp_share_post2014 = 0.95   # post-2014 (Indian Express / SC plea)

# BASE-RATE BENCHMARK: what share of politicians are 'opposition'?
# The 14-party plea itself states the petitioning parties represented ~45.2% of
# votes in the last state polls and 42.5% in the 2019 general election.
# A neutral agency's caseload should roughly track the share of politicians who
# are in opposition. If ~half of politically-relevant actors are opposition,
# a neutral base rate is ~50-55%. Post-2014's 95% is the anomaly to explain.
neutral_benchmark = 0.55
lift_pre = opp_share_pre2014/neutral_benchmark
lift_post = opp_share_post2014/neutral_benchmark
print(f"  Neutral base-rate benchmark (opposition share of actors): ~{neutral_benchmark*100:.0f}%")
print(f"  Pre-2014 opposition share of probes:  {opp_share_pre2014*100:.0f}%  (lift {lift_pre:.2f}x)")
print(f"  Post-2014 opposition share of probes: {opp_share_post2014*100:.0f}%  (lift {lift_post:.2f}x)")
print(f"  Shift in opposition share: +{(opp_share_post2014-opp_share_pre2014)*100:.0f} pts")

# Odds-ratio framing (how the odds of 'being opposition | investigated' changed)
def odds(pr): return pr/(1-pr)
or_shift = odds(opp_share_post2014)/odds(opp_share_pre2014)
print(f"  Odds ratio (post vs pre-2014): {or_shift:.1f}x")
print(f"    -> The odds that an investigated politician is opposition rose ~{or_shift:.0f}-fold.")
print(f"    Pre-2014 was already above a neutral base rate (incumbency effects exist),")
print(f"    but the post-2014 jump to 95% is far beyond what incumbency alone explains.")

# ---------------------------------------------------------------
# PROPOSITION 3 (bridge to money): does the enforcement asymmetry co-move with
# the funding asymmetry? Electoral bonds = court-disclosed data (VERIFIED).
# ---------------------------------------------------------------
print("\n--- Proposition 3: Does enforcement asymmetry co-move with funding capture? ---")
bjp_bond_share = 6566/13558      # ECI data via Tribune: Rs6,566cr of Rs13,558cr
print(f"  BJP share of electoral-bond money (20 parties): {bjp_bond_share*100:.1f}%")
print(f"  Opposition share of ED/CBI probes (post-2014):  {opp_share_post2014*100:.0f}%")
print(f"  The party receiving ~48% of all opaque bond funding is the same party")
print(f"    whose rivals absorb ~95% of coercive enforcement. These are two")
print(f"    independently-sourced datasets pointing the same direction: resources")
print(f"    flow toward power, coercion flows away from it. That co-movement is")
print(f"    the quantitative signature of the mission thesis at the accountability layer.")

print("\n" + "="*70)
print("STUDY C VERDICT")
print("="*70)
print("""  P1 (decoupling): SUPPORTED, robustly, on VERIFIED govt data.
     A coercive process with a ~1% conviction rate is, functionally, a
     process whose output is the process itself.
  P2 (asymmetry): SUPPORTED, but on WEAKER journalistic/litigant data,
     because the govt refuses to publish the party-wise breakdown. The
     refusal to publish is itself a finding.
  P3 (co-movement): SUGGESTIVE. Two independent public datasets (bond
     disclosure + enforcement counts) align with the thesis.

  WHAT WOULD FALSIFY IT: case-level data showing (a) opposition share of
  probes tracking their share of political actors, or (b) conviction rates
  on politician cases comparable to non-political PMLA cases. The government
  holds this data and has not released it.

  PUBLISHABILITY: P1 alone is a clean, defensible short paper/blog artifact
  built entirely on the state's own numbers. P2/P3 need the microdata to
  move from 'suggestive' to 'demonstrated'.""")
