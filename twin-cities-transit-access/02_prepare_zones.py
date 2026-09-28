"""Build block-group origins and destinations from 2020 blocks + LODES.

AO routes from every Census block centroid. This scaled-down version routes
between block groups (BGs), but places each BG's point where its people or
jobs actually are rather than at the polygon centroid:

  origin point      = worker-weighted mean of block internal points (LODES RAC)
  destination point = job-weighted mean of block internal points (LODES WAC)

Outputs (data/interim/):
  origins.parquet        BG, workers, pop, geometry (origin point)
  destinations.parquet   BG, jobs, geometry (destination point)
  blocks_jobs.parquet    block-level job points (for the aggregation check)
  bg_polygons.parquet    BG polygons for mapping
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from config import COUNTIES, INTERIM, RAW, STATE_FIPS


def weighted_points(df, weight, fallback_weight=None):
    """Per-BG weighted mean of block internal points."""
    w = df[weight].astype(float)
    if fallback_weight is not None:
        bg_tot = w.groupby(df["bg"]).transform("sum")
        w = np.where(bg_tot > 0, w, df[fallback_weight].astype(float))
    d = df.assign(_w=w, _wx=w * df["x"], _wy=w * df["y"])
    g = d.groupby("bg")[["_w", "_wx", "_wy"]].sum()
    g = g[g["_w"] > 0]
    return gpd.GeoDataFrame(
        index=g.index,
        geometry=gpd.points_from_xy(g["_wx"] / g["_w"], g["_wy"] / g["_w"]),
        crs="EPSG:4326",
    )


def main():
    lodes_year = json.loads((RAW / "MANIFEST.json").read_text())["lodes_year"]

    blocks = gpd.read_file(
        RAW / "tl_2020_27_tabblock20.zip",
        columns=["GEOID20", "COUNTYFP20", "ALAND20", "POP20", "INTPTLAT20", "INTPTLON20"],
        ignore_geometry=True,
    )
    blocks = blocks[blocks["COUNTYFP20"].isin(COUNTIES)].copy()
    blocks["bg"] = blocks["GEOID20"].str[:12]
    blocks["x"] = blocks["INTPTLON20"].astype(float)
    blocks["y"] = blocks["INTPTLAT20"].astype(float)

    wac = pd.read_csv(RAW / f"mn_wac_S000_JT00_{lodes_year}.csv.gz",
                      usecols=["w_geocode", "C000"], dtype={"w_geocode": str})
    rac = pd.read_csv(RAW / f"mn_rac_S000_JT00_{lodes_year}.csv.gz",
                      usecols=["h_geocode", "C000"], dtype={"h_geocode": str})
    blocks = (blocks
              .merge(wac.rename(columns={"w_geocode": "GEOID20", "C000": "jobs"}), how="left")
              .merge(rac.rename(columns={"h_geocode": "GEOID20", "C000": "workers"}), how="left"))
    blocks[["jobs", "workers"]] = blocks[["jobs", "workers"]].fillna(0).astype(int)

    # LODES blocks that failed to match a 2020 block in the study area would be
    # silently dropped; report how many in-area jobs/workers survived the join.
    in_area = lambda s: s.str[:5].isin({STATE_FIPS + c for c in COUNTIES})
    lodes_jobs = wac.loc[in_area(wac["w_geocode"]), "C000"].sum()
    lodes_workers = rac.loc[in_area(rac["h_geocode"]), "C000"].sum()

    bg_stats = blocks.groupby("bg").agg(jobs=("jobs", "sum"), workers=("workers", "sum"),
                                        pop=("POP20", "sum"))

    origins = weighted_points(blocks, "workers", fallback_weight="POP20").join(bg_stats)
    origins = origins[origins["workers"] > 0]
    dests = weighted_points(blocks[blocks["jobs"] > 0], "jobs").join(bg_stats[["jobs"]])

    for g in (origins, dests):
        g.index.name = "id"
    origins.reset_index().to_parquet(INTERIM / "origins.parquet")
    dests.reset_index().to_parquet(INTERIM / "destinations.parquet")

    bj = blocks[blocks["jobs"] > 0]
    gpd.GeoDataFrame(bj[["GEOID20", "bg", "jobs"]].rename(columns={"GEOID20": "id"}),
                     geometry=gpd.points_from_xy(bj["x"], bj["y"]), crs="EPSG:4326"
                     ).to_parquet(INTERIM / "blocks_jobs.parquet")

    bgs = gpd.read_file(RAW / "tl_2020_27_bg.zip", columns=["GEOID", "COUNTYFP", "ALAND"])
    bgs = bgs[bgs["COUNTYFP"].isin(COUNTIES)].rename(columns={"GEOID": "id"})
    bgs.to_crs("EPSG:4326").to_parquet(INTERIM / "bg_polygons.parquet")

    summary = {
        "lodes_year": lodes_year,
        "block_groups_total": int(len(bgs)),
        "origin_bgs_with_workers": int(len(origins)),
        "destination_bgs_with_jobs": int(len(dests)),
        "blocks_with_jobs": int(len(bj)),
        "jobs_total": int(dests["jobs"].sum()),
        "workers_total": int(origins["workers"].sum()),
        "lodes_jobs_in_counties": int(lodes_jobs),
        "lodes_workers_in_counties": int(lodes_workers),
    }
    (INTERIM / "zones_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
