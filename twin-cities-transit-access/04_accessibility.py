"""Turn per-departure cumulative curves into AO-style accessibility results.

Location measure (per origin BG, per threshold t):
    transit: mean over departure minutes 07:00-08:59 of jobs reachable within t
    walk:    jobs reachable within t (walking is not time-dependent)
Person measure (region): mean of the location measure weighted by LODES RAC
    workers living in each BG (AO "worker-weighted accessibility").

Writes outputs/bg_accessibility.csv, outputs/curve.csv, outputs/summary.json.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from config import AO_BETA, COARSE_STEP_MIN, COUNTIES, HEADLINE, INTERIM, OUT, THRESHOLDS

ACCESS = INTERIM / "access"


def worker_weighted(values, w):
    return float(np.average(values, weights=w))


def ao_weighted_score(curve_by_t):
    """AO ranking metric: sum over 10-min 'donuts' of added jobs x exp(beta t)."""
    ts = range(10, 61, 10)
    return float(sum((curve_by_t[t] - curve_by_t.get(t - 10, 0.0)) * np.exp(AO_BETA * t) for t in ts))


def main():
    origins = gpd.read_parquet(INTERIM / "origins.parquet")
    zones = json.loads((INTERIM / "zones_summary.json").read_text())
    w = origins["workers"].to_numpy()
    walk = np.load(ACCESS / "walk.npy")

    files = sorted(ACCESS.glob("transit_*.npy"))
    tags = [f.stem.split("_")[1] for f in files]
    stack = np.stack([np.load(f) for f in files])            # (n_dep, n_orig, 61)
    minute = np.array([(int(t[:2]) - 7) * 60 + int(t[2:]) for t in tags])
    transit = stack.mean(axis=0)
    coarse = stack[minute % COARSE_STEP_MIN == 0].mean(axis=0)

    # ---- per-BG table --------------------------------------------------------
    bg = pd.DataFrame({
        "bg_geoid": origins["id"],
        "county": origins["id"].str[2:5].map(COUNTIES),
        "workers": w,
        "jobs_in_bg": origins["id"].map(
            gpd.read_parquet(INTERIM / "destinations.parquet").set_index("id")["jobs"]).fillna(0).astype(int),
    })
    for t in HEADLINE:
        bg[f"walk_{t}"] = walk[:, t]
        bg[f"transit_{t}"] = transit[:, t].round(0)
        # Departure-time sensitivity: worst and best single minute in the window.
        bg[f"transit_{t}_min"] = stack[:, :, t].min(axis=0)
        bg[f"transit_{t}_max"] = stack[:, :, t].max(axis=0)
        bg[f"ratio_{t}"] = (transit[:, t] / np.maximum(walk[:, t], 1)).round(2)
    bg.to_csv(OUT / "bg_accessibility.csv", index=False)

    # ---- worker-weighted curves, t = 0..60 ------------------------------------
    curve = pd.DataFrame({
        "minutes": np.arange(61),
        "transit": [worker_weighted(transit[:, t], w) for t in range(61)],
        "walk": [worker_weighted(walk[:, t], w) for t in range(61)],
    })
    curve.to_csv(OUT / "curve.csv", index=False, float_format="%.1f")
    cw = dict(zip(curve["minutes"], curve["walk"]))
    ct = dict(zip(curve["minutes"], curve["transit"]))

    # ---- headline numbers ------------------------------------------------------
    total_jobs = zones["jobs_total"]
    s = {
        "lodes_year": zones["lodes_year"],
        "departures_used": len(files),
        "jobs_total": total_jobs,
        "workers_total": int(w.sum()),
        "origin_bgs": int(len(origins)),
        "blocks_with_jobs": zones["blocks_with_jobs"],
        "ao_weighted_score": {"transit": ao_weighted_score(ct), "walk": ao_weighted_score(cw)},
        "thresholds": {},
    }
    for t in THRESHOLDS:
        tr, wk = ct[t], cw[t]
        tr_coarse = worker_weighted(coarse[:, t], w)
        s["thresholds"][t] = {
            "transit_ww": round(tr), "walk_ww": round(wk),
            "transit_share_of_jobs": round(tr / total_jobs, 4),
            "walk_share_of_jobs": round(wk / total_jobs, 4),
            "transit_over_walk": round(tr / wk, 2),
            "transit_ww_5min_sample": round(tr_coarse),
            "sampling_diff_pct": round(100 * (tr_coarse - tr) / tr, 2),
        }

    # Distributional facts for the findings (at the headline thresholds).
    for t in HEADLINE:
        tr_t = transit[:, t]
        order = np.argsort(tr_t)[::-1]
        cumw = np.cumsum(w[order]) / w.sum()
        gain = tr_t - walk[:, t]
        swing = stack[:, :, t].max(axis=0) - stack[:, :, t].min(axis=0)
        by_county = (bg.assign(tr=tr_t, wk=walk[:, t])
                     .groupby("county")
                     .apply(lambda d: pd.Series({
                         "workers": int(d["workers"].sum()),
                         "transit_ww": round(np.average(d["tr"], weights=d["workers"])),
                         "walk_ww": round(np.average(d["wk"], weights=d["workers"])),
                     }), include_groups=False)
                     .sort_values("transit_ww", ascending=False))
        s[f"dist_{t}"] = {
            "median_bg_transit": round(float(np.median(tr_t))),
            "median_bg_walk": round(float(np.median(walk[:, t]))),
            "worker_weighted_median_transit": round(float(tr_t[order][np.searchsorted(cumw, 0.5)])),
            "share_workers_transit_ge_10pct_jobs": round(float(w[tr_t >= 0.10 * total_jobs].sum() / w.sum()), 4),
            "share_workers_transit_lt_1pct_jobs": round(float(w[tr_t < 0.01 * total_jobs].sum() / w.sum()), 4),
            "share_workers_transit_gain_lt_2x": round(float(w[tr_t < 2 * np.maximum(walk[:, t], 1)].sum() / w.sum()), 4),
            "share_workers_in_top_decile_bgs": round(float(w[order[: len(order) // 10]].sum() / w.sum()), 4),
            "median_bg_departure_swing": round(float(np.median(swing))),
            "median_bg_swing_pct_of_mean": round(float(np.median(swing / np.maximum(tr_t, 1))) * 100, 1),
            "worker_weighted_gain": round(float(np.average(gain, weights=w))),
            "by_county": by_county.reset_index().to_dict(orient="records"),
        }

    # Aggregation check: walk to block-level destinations vs BG points.
    blk = ACCESS / "walk_blockdest.npy"
    if blk.exists():
        wb = np.load(blk)
        s["aggregation_check_walk"] = {
            t: {"bg_dest_ww": round(cw[t]), "block_dest_ww": round(worker_weighted(wb[:, t], w)),
                "diff_pct": round(100 * (cw[t] - worker_weighted(wb[:, t], w)) / worker_weighted(wb[:, t], w), 2)}
            for t in HEADLINE
        }

    (OUT / "summary.json").write_text(json.dumps(s, indent=2, default=int))
    print(json.dumps({k: v for k, v in s.items() if not k.startswith("dist")}, indent=2, default=int))
    for t in HEADLINE:
        d = dict(s[f"dist_{t}"]); d.pop("by_county")
        print(t, json.dumps(d, indent=1))
        print(pd.DataFrame(s[f"dist_{t}"]["by_county"]).to_string(index=False))


if __name__ == "__main__":
    main()
