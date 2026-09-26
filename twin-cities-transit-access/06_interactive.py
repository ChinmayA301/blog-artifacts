"""Build the self-contained interactive map from the template.

Writes index.html (served by GitHub Pages) from interactive_template.html.
All data is inlined, so the page makes no requests beyond Leaflet and fonts.
"""
import json

import geopandas as gpd
import pandas as pd
import requests

from config import INTERIM, OUT, ROOT
from narrative import AO_2024, findings, methods_and_limits

LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"
SIMPLIFY_M = 25  # metres; block-group edges stay recognisable at metro zoom


def geojson(gdf, precision=5):
    return json.loads(gdf.to_json(drop_id=True, to_wgs84=True))


def round_coords(obj, nd=5):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, list):
        return [round_coords(v, nd) for v in obj]
    if isinstance(obj, dict):
        return {k: round_coords(v, nd) for k, v in obj.items()}
    return obj


def main():
    s = json.loads((OUT / "summary.json").read_text())
    acc = pd.read_csv(OUT / "bg_accessibility.csv", dtype={"bg_geoid": str})
    bg = gpd.read_parquet(INTERIM / "bg_polygons.parquet").to_crs(26915)
    bg["geometry"] = bg.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    bg = bg.merge(acc, left_on="id", right_on="bg_geoid", how="left")
    props = bg.assign(
        t30=bg["transit_30"], t45=bg["transit_45"], w30=bg["walk_30"], w45=bg["walk_45"],
        r30=bg["ratio_30"], r45=bg["ratio_45"], t45min=bg["transit_45_min"], t45max=bg["transit_45_max"],
        county=bg["county"].fillna(""), workers=bg["workers"].fillna(0),
    )[["id", "county", "workers", "t30", "t45", "w30", "w45", "r30", "r45", "t45min", "t45max", "geometry"]]
    # Block groups with no resident workers have no origin; they render as empty.
    for c in ["t30", "t45", "w30", "w45", "t45min", "t45max", "workers"]:
        props[c] = props[c].astype("Int64")
    counties = bg.dissolve("COUNTYFP")[["geometry"]].reset_index(drop=True)

    import importlib
    rail = importlib.import_module("05_onepager").rail_lines().to_crs(26915)
    rail["geometry"] = rail.geometry.simplify(10)

    curve = pd.read_csv(OUT / "curve.csv")
    method, limits = methods_and_limits(s)
    data = {
        "summary": s,
        "curve": {"transit": curve["transit"].round().tolist(), "walk": curve["walk"].round().tolist()},
        "ao": AO_2024,
        "findings": findings(s),
        "methods": method + limits,
        "bg": round_coords(geojson(props)),
        "counties": round_coords(geojson(counties)),
        "rail": round_coords(geojson(rail)),
    }
    blob = json.dumps(data, separators=(",", ":"), default=int).replace("</", "<\\/")

    tpl = (ROOT / "interactive_template.html").read_text()
    css = requests.get(LEAFLET_CSS, timeout=30).text
    body = tpl.replace("/*__LEAFLET_CSS__*/", css).replace("/*__DATA__*/null", blob)
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + body.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1)
            + "\n</body>\n</html>\n")
    (ROOT / "index.html").write_text(page)
    print(f"index.html {len(page) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
