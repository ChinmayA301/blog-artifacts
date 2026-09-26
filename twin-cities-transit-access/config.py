"""Shared paths and parameters for the Twin Cities transit-access replication.

Every parameter that mirrors the Accessibility Observatory (AO) method is
tagged `# AO:` with the source; every deliberate deviation is tagged
`# DEVIATION:` and repeated in the README's methods-and-limits table.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# TCA_DATA moves the ~600 MB of raw + intermediate data out of the repo, e.g. when
# the checkout lives in an iCloud-synced folder that evicts files under disk pressure.
DATA = Path(os.environ.get("TCA_DATA", ROOT / "data")).expanduser()
RAW = DATA / "raw"
INTERIM = DATA / "interim"
OUT = ROOT / "outputs"
for d in (RAW, INTERIM, OUT):
    d.mkdir(parents=True, exist_ok=True)

# r5py needs a JDK 21. Point JAVA_HOME at one before importing r5py.
os.environ.setdefault(
    "JAVA_HOME",
    next((str(p) for p in (Path.home() / ".local" / "jdk").glob("jdk-21*/Contents/Home")), ""),
)

# --- Study area -------------------------------------------------------------
# The seven counties of the Metropolitan Council region, i.e. Metro Transit's
# planning area. DEVIATION: AO aggregates to the full Minneapolis-St. Paul CBSA
# (16 counties incl. two in Wisconsin); jobs outside these seven are excluded.
STATE_FIPS = "27"
COUNTIES = {
    "003": "Anoka", "019": "Carver", "037": "Dakota", "053": "Hennepin",
    "123": "Ramsey", "139": "Scott", "163": "Washington",
}
# OSM clip box (lon/lat), padded ~5 km beyond the seven-county extent.
BBOX = (-94.07, 44.41, -92.67, 45.48)

# --- Sources ----------------------------------------------------------------
GTFS_URL = "https://svc.metrotransit.org/mtgtfs/gtfs.zip"
OSM_URL = "https://download.geofabrik.de/north-america/us/minnesota-latest.osm.pbf"
TIGER_BLOCKS_URL = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_27_tabblock20.zip"
TIGER_BG_URL = "https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_27_bg.zip"
LODES_BASE = "https://lehd.ces.census.gov/data/lodes/LODES8/mn"
LODES_YEARS_TO_TRY = (2024, 2023, 2022)  # newest available wins

# --- Routing parameters -----------------------------------------------------
WALK_SPEED_KMH = 3.6          # AO 2022 methodology, s.3.3.2 and s.3.3.4
DEPARTURE_START = "07:00"     # AO Transit 2014 methodology, s.4.4: every minute 07:00-08:59
DEPARTURE_MINUTES = 120
# Every minute is routed, as AO does. Multiples of COARSE_STEP_MIN run first so a
# usable 5-minute sample exists early; 04_accessibility.py reports both.
COARSE_STEP_MIN = 5
MAX_TIME_MIN = 60             # AO thresholds run to 60 minutes
THRESHOLDS = (10, 20, 30, 40, 45, 50, 60)   # 30 and 45 are the headline thresholds
HEADLINE = (30, 45)
AO_BETA = -0.08               # AO weighted-ranking decay (2022 methodology, s.3.6)
