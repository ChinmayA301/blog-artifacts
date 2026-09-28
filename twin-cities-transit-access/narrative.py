"""Findings and methods-and-limits text, shared by the PDF and the interactive page.

Every number is read from outputs/summary.json; nothing is typed by hand.
"""

# AO, Access Across America: Transit 2024 (CTS 25-18), Minneapolis profile, Table 2.
AO_2024 = {10: 409, 20: 3737, 30: 15664, 40: 39282, 50: 74945, 60: 116103}


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}%"


def findings(s):
    t30, t45 = s["thresholds"]["30"], s["thresholds"]["45"]
    d30, d45 = s["dist_30"], s["dist_45"]
    core = {r["county"]: r for r in d30["by_county"]}
    suburban = [r for r in d30["by_county"] if r["county"] not in ("Hennepin", "Ramsey")]
    sub_w = sum(r["workers"] for r in suburban)
    sub_tr = sum(r["transit_ww"] * r["workers"] for r in suburban) / sub_w
    sub_wk = sum(r["walk_ww"] * r["workers"] for r in suburban) / sub_w
    return [
        ("Transit multiplies reach, but from a small base.",
         f"The average worker can reach {t30['transit_ww']:,} jobs within 30 minutes by transit and "
         f"{t45['transit_ww']:,} within 45, about {t30['transit_over_walk']:.1f}x and "
         f"{t45['transit_over_walk']:.1f}x what walking alone reaches. Even at 45 minutes that is "
         f"{pct(t45['transit_share_of_jobs'], 1)} of the region's {s['jobs_total'] / 1e6:.2f} million jobs."),
        ("The gain is concentrated in the two core counties.",
         f"Hennepin and Ramsey workers reach {core['Hennepin']['transit_ww']:,} and "
         f"{core['Ramsey']['transit_ww']:,} jobs in 30 minutes by transit. In the five other counties "
         f"the figure is {sub_tr:,.0f}, against {sub_wk:,.0f} on foot. For "
         f"{pct(d30['share_workers_transit_gain_lt_2x'])} of all workers, transit does not even double "
         f"their 30-minute walking reach."),
        ("Departure minute matters, not just location.",
         f"For the median block group, 45-minute transit access swings by "
         f"{d45['median_bg_swing_pct_of_mean']:.0f}% of its average between the best and worst departure "
         f"minute in 7-9 AM. That is service frequency showing up in the numbers, which is why AO averages "
         f"every minute instead of picking one."),
    ]


def methods_and_limits(s):
    """Returns (method_rows, limit_rows), each a list of (heading, body)."""
    agg = s.get("aggregation_check_walk", {})
    t30 = s["thresholds"]["30"]
    max_gap = max(abs(s["thresholds"][str(t)]["transit_ww"] / AO_2024[t] - 1) * 100 for t in (30, 40, 50, 60))
    method = [
        ("Measure", "Cumulative opportunities: jobs reachable within t minutes, averaged over every departure "
                    f"minute 7:00–8:59 AM ({s['departures_used']} routed departures), then weighted by the "
                    "workers living in each origin (AO's worker-weighted accessibility)."),
        ("Routing", "r5py (the R5 engine AO also uses) on OpenStreetMap + Metro Transit GTFS. Walk 3.6 km/h "
                    "for access, egress and transfers; unlimited transfers; a walk-only trip counts as a transit "
                    "trip if faster, as in AO."),
        ("Data", f"Jobs: LEHD LODES8 {s['lodes_year']} WAC. Workers: LODES8 RAC. Geography: 2020 TIGER blocks "
                 "rolled up to block groups; origin and destination points are worker- and job-weighted block "
                 "centroids, not polygon centroids."),
        ("Check", f"Against AO's published Transit 2024 Twin Cities figures (chart, open circles): "
                  f"{t30['transit_ww']:,} vs {AO_2024[30]:,} at 30 min, and within {max_gap:.0f}% at every "
                  "threshold from 30 to 60 min. The differences below partly offset each other, so read "
                  "this as a sanity check, not a validation."),
        ("Open question", "AO's Transit 2024 abstract describes the 15% fastest travel times; its body "
                          "describes an average over 7-9 AM. This run uses the average. Which one produces "
                          "the headline figures?"),
    ]
    limits = [
        ("Coarser zones", f"AO routes from every Census block; this uses {s['origin_bgs']:,} block groups. Jobs inside a "
                          "worker's own block group count as a few minutes away, so the 10- and 20-minute values "
                          f"run high (about {s['thresholds']['10']['transit_ww'] / AO_2024[10]:.1f}x AO at 10 min). "
                          "Read 30 and 45 min, not below."),
        ("Smaller region", "Seven counties, not AO's 16-county CBSA (which includes two Wisconsin counties). Jobs "
                           "just outside the boundary are invisible, which understates access near the edge."),
        ("Walk access only", "Like AO, nobody drives to a stop. SouthWest Transit's park-and-ride expresses "
                             "are in the feed but add nothing to Carver County at 45 min. MVTA is not in the "
                             "feed, so Scott shows no transit gain at all and Dakota is understated."),
        ("Schedules, not reality", "Perfect on-time running, one fall 2026 Wednesday. LODES counts jobs where "
                                   "they are registered, and it cannot see remote work."),
    ]
    if agg:
        limits.append(("Aggregation test", f"Walking re-run to {s['blocks_with_jobs']:,} block-level job points: "
                                           "block-group destinations shift 30-min access by "
                                           f"{agg['30']['diff_pct']:+.1f}% and 45-min by {agg['45']['diff_pct']:+.1f}%."))
    return method, limits
