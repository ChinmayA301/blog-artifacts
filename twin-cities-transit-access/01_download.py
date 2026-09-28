"""Download raw inputs: Metro Transit GTFS, OSM, TIGER 2020 blocks/BGs, LODES8 WAC+RAC.

Then clip the Minnesota OSM extract to the study bbox with osmium-tool.
Re-running skips files already present.
"""
import datetime as dt
import json
import shutil
import subprocess

import requests

from config import (BBOX, GTFS_URL, LODES_BASE, LODES_YEARS_TO_TRY, OSM_URL, RAW,
                    TIGER_BG_URL, TIGER_BLOCKS_URL)


def fetch(url, dest):
    if dest.exists():
        print(f"  have {dest.name}")
        return
    print(f"  get  {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)


def latest_lodes_year():
    for y in LODES_YEARS_TO_TRY:
        url = f"{LODES_BASE}/wac/mn_wac_S000_JT00_{y}.csv.gz"
        if requests.head(url, timeout=30).status_code == 200:
            return y
    raise RuntimeError("no LODES8 WAC file found for the configured years")


def main():
    manifest = {"retrieved": dt.date.today().isoformat(), "files": {}}

    fetch(GTFS_URL, RAW / "metrotransit_gtfs.zip")
    manifest["files"]["metrotransit_gtfs.zip"] = GTFS_URL
    fetch(TIGER_BLOCKS_URL, RAW / "tl_2020_27_tabblock20.zip")
    manifest["files"]["tl_2020_27_tabblock20.zip"] = TIGER_BLOCKS_URL
    fetch(TIGER_BG_URL, RAW / "tl_2020_27_bg.zip")
    manifest["files"]["tl_2020_27_bg.zip"] = TIGER_BG_URL

    year = latest_lodes_year()
    manifest["lodes_year"] = year
    for kind in ("wac", "rac"):
        name = f"mn_{kind}_S000_JT00_{year}.csv.gz"
        fetch(f"{LODES_BASE}/{kind}/{name}", RAW / name)
        manifest["files"][name] = f"{LODES_BASE}/{kind}/{name}"

    mn = RAW / "minnesota-latest.osm.pbf"
    clipped = RAW / "twin_cities.osm.pbf"
    if not clipped.exists():
        fetch(OSM_URL, mn)
        if not shutil.which("osmium"):
            raise SystemExit("osmium-tool not found (brew install osmium-tool)")
        bbox = ",".join(str(v) for v in BBOX)
        subprocess.run(["osmium", "extract", "-b", bbox, "-s", "complete_ways",
                        str(mn), "-o", str(clipped), "--overwrite"], check=True)
        mn.unlink()  # the statewide file is ~290 MB and no longer needed
    manifest["files"]["twin_cities.osm.pbf"] = f"{OSM_URL} clipped to bbox {BBOX}"

    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
