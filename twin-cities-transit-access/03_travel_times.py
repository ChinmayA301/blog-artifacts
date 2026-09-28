"""Route BG->BG travel times with r5py and reduce each run to cumulative job counts.

For every origin block group and every minute budget t = 0..60, we store the
number of jobs reachable within t minutes ("cumulative opportunities").
Raw OD matrices are not kept (2.5k x 2.6k per departure); only the reduced
per-origin curves are written, one .npy per departure time:

  data/interim/access/walk.npy               shape (n_origins, 61)
  data/interim/access/transit_HHMM.npy       shape (n_origins, 61)

Usage:
  python 03_travel_times.py walk
  python 03_travel_times.py transit            # all departures, skips done ones
  python 03_travel_times.py transit --only 0700  # a single departure (benchmark)
  python 03_travel_times.py blockcheck         # walk to block-level destinations
"""
import argparse
import datetime as dt
import time
import warnings

import geopandas as gpd
import numpy as np

import config  # noqa: F401  (sets JAVA_HOME before r5py import)
from config import (DEPARTURE_MINUTES, DEPARTURE_START, COARSE_STEP_MIN, INTERIM,
                    MAX_TIME_MIN, RAW, WALK_SPEED_KMH)

# A Wednesday with normal weekday service inside the feed's validity window
# (feed valid 2026-09-19..2026-12-04; Wednesdays run their own service_id).
# AO: "a Wednesday with normal, non-holiday service" (2022 methodology, s.2.5).
SERVICE_DATE = dt.date(2026, 10, 7)
ACCESS = INTERIM / "access"
ACCESS.mkdir(exist_ok=True)


def load_network():
    import r5py
    return r5py.TransportNetwork(RAW / "twin_cities.osm.pbf", [RAW / "metrotransit_gtfs.zip"])


def cumulative_jobs(ttm, origin_ids, dest_jobs):
    """(n_origins, MAX_TIME_MIN+1) array: jobs reachable within t minutes."""
    df = ttm.dropna(subset=["travel_time"])
    oi = origin_ids.get_indexer(df["from_id"])
    tt = df["travel_time"].to_numpy().astype(int)
    jobs = dest_jobs.reindex(df["to_id"]).to_numpy()
    hist = np.zeros((len(origin_ids), MAX_TIME_MIN + 1), dtype=np.int64)
    np.add.at(hist, (oi, np.clip(tt, 0, MAX_TIME_MIN)), jobs)
    return np.cumsum(hist, axis=1)


def route(network, origins, dests, modes, departure, window):
    import r5py
    with warnings.catch_warnings():
        # A 1-minute window is intentional: we want exactly one departure minute.
        warnings.filterwarnings("ignore", message="The provided departure time window")
        return r5py.TravelTimeMatrix(
            network,
            origins=origins,
            destinations=dests,
            departure=departure,
            departure_time_window=window,
            transport_modes=modes,
            max_time=dt.timedelta(minutes=MAX_TIME_MIN),
            speed_walking=WALK_SPEED_KMH,
            max_public_transport_rides=8,
            percentiles=[50],
            snap_to_network=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["walk", "transit", "blockcheck"])
    ap.add_argument("--only", help="HHMM of a single transit departure")
    args = ap.parse_args()

    import r5py
    origins = gpd.read_parquet(INTERIM / "origins.parquet")
    dests = gpd.read_parquet(INTERIM / "destinations.parquet")
    origin_ids = origins.set_index("id").index
    t0 = time.time()
    network = load_network()
    print(f"network ready in {time.time() - t0:.0f}s")

    start = dt.datetime.combine(SERVICE_DATE, dt.time.fromisoformat(DEPARTURE_START))
    one_min = dt.timedelta(minutes=1)

    if args.mode == "walk":
        ttm = route(network, origins, dests, [r5py.TransportMode.WALK], start, one_min)
        np.save(ACCESS / "walk.npy", cumulative_jobs(ttm, origin_ids, dests.set_index("id")["jobs"]))
        print(f"walk done in {time.time() - t0:.0f}s")
        return

    if args.mode == "blockcheck":
        # Same walk run, but to 19k block-level job points instead of BG points,
        # to measure how much aggregating destinations to BGs moves the answer.
        blocks = gpd.read_parquet(INTERIM / "blocks_jobs.parquet")
        ttm = route(network, origins, blocks, [r5py.TransportMode.WALK], start, one_min)
        np.save(ACCESS / "walk_blockdest.npy",
                cumulative_jobs(ttm, origin_ids, blocks.set_index("id")["jobs"]))
        print(f"blockcheck done in {time.time() - t0:.0f}s")
        return

    modes = [r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK]
    # Coarse 5-minute departures first, then fill in the remaining minutes.
    minutes = sorted(range(DEPARTURE_MINUTES), key=lambda m: (m % COARSE_STEP_MIN != 0, m))
    for m in minutes:
        dep = start + dt.timedelta(minutes=m)
        tag = dep.strftime("%H%M")
        if args.only and tag != args.only:
            continue
        out = ACCESS / f"transit_{tag}.npy"
        if out.exists():
            continue
        t1 = time.time()
        ttm = route(network, origins, dests, modes, dep, one_min)
        np.save(out, cumulative_jobs(ttm, origin_ids, dests.set_index("id")["jobs"]))
        print(f"transit {tag}: {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
