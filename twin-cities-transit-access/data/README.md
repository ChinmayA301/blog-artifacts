# Data

Nothing in `data/` is committed. `01_download.py` fetches everything and writes
`raw/MANIFEST.json` with the exact URLs and retrieval date.

| File | Source | Used for |
|---|---|---|
| `raw/metrotransit_gtfs.zip` | Metro Transit / Metropolitan Council GTFS, <https://svc.metrotransit.org/mtgtfs/gtfs.zip> | Transit schedules. The run uses Wednesday 7 Oct 2026, 07:00–08:59. The feed also covers Maple Grove, Plymouth, SouthWest Transit, U of M and MSP airport routes. MVTA is **not** included. |
| `raw/twin_cities.osm.pbf` | Geofabrik Minnesota extract, clipped to the study bbox with `osmium extract` | Walking network |
| `raw/tl_2020_27_tabblock20.zip` | Census TIGER/Line 2020 blocks | Block internal points and 2020 population |
| `raw/tl_2020_27_bg.zip` | Census TIGER/Line 2020 block groups | Map polygons |
| `raw/mn_wac_S000_JT00_<year>.csv.gz` | LEHD LODES8 Workplace Area Characteristics | Jobs by block (destinations) |
| `raw/mn_rac_S000_JT00_<year>.csv.gz` | LEHD LODES8 Residence Area Characteristics | Resident workers by block (origin weights) |

`interim/` holds the block-group origin/destination points (`02_prepare_zones.py`) and the
per-departure cumulative job curves in `interim/access/` (`03_travel_times.py`).
Raw OD matrices are never written to disk.
