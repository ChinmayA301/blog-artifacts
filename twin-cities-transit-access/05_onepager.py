"""Render the one-page PDF: map, one chart, three findings, methods-and-limits box.

Every number in the text is read from outputs/summary.json. Nothing is typed by hand.
"""
import io
import json
import textwrap
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from shapely.geometry import LineString

from config import INTERIM, OUT, RAW
from narrative import AO_2024, findings, methods_and_limits

INTERACTIVE_URL = "https://chinmaya301.github.io/blog-artifacts/twin-cities-transit-access/"
CODE_URL = "https://github.com/ChinmayA301/blog-artifacts/tree/main/twin-cities-transit-access"


INK, INK2, MUTED, RULE, PANEL = "#16201c", "#45514b", "#6b7670", "#d5dad6", "#f3f5f2"
TRANSIT, WALK = "#2a78d6", "#eb6834"
RAMP = ["#e3eefb", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
BINS = [0, 1_000, 5_000, 25_000, 100_000, 250_000, 2_000_000]
BIN_LABELS = ["< 1k", "1k–5k", "5k–25k", "25k–100k", "100k–250k", "250k +"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "text.color": INK,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "pdf.fonttype": 42,
})


def k(n):
    return f"{n / 1000:,.1f}k" if n < 10_000 else f"{n / 1000:,.0f}k"


def rail_lines():
    """METRO Blue and Green light-rail alignments from the GTFS shapes."""
    with zipfile.ZipFile(RAW / "metrotransit_gtfs.zip") as z:
        trips = pd.read_csv(z.open("trips.txt"), dtype=str, usecols=["route_id", "shape_id"])
        shapes = pd.read_csv(z.open("shapes.txt"), dtype={"shape_id": str})
    ids = trips[trips["route_id"].isin(["901", "902"])].drop_duplicates("route_id")
    out = []
    for rid, sid in zip(ids["route_id"], ids["shape_id"]):
        s = shapes[shapes["shape_id"] == sid].sort_values("shape_pt_sequence")
        out.append({"route": rid, "geometry": LineString(zip(s["shape_pt_lon"], s["shape_pt_lat"]))})
    return gpd.GeoDataFrame(out, crs="EPSG:4326")


def draw_map(ax, s):
    bg = gpd.read_parquet(INTERIM / "bg_polygons.parquet")
    acc = pd.read_csv(OUT / "bg_accessibility.csv", dtype={"bg_geoid": str})
    bg = bg.merge(acc, left_on="id", right_on="bg_geoid", how="left").to_crs(26915)
    counties = bg.dissolve("COUNTYFP").boundary
    cmap = ListedColormap(RAMP)
    norm = BoundaryNorm(BINS, cmap.N)
    bg[bg["transit_45"].isna()].plot(ax=ax, color="#eceeeb", linewidth=0)
    bg[bg["transit_45"].notna()].plot(ax=ax, column="transit_45", cmap=cmap, norm=norm, linewidth=0)
    counties.plot(ax=ax, color="white", linewidth=0.6)
    rail_lines().to_crs(26915).plot(ax=ax, color=INK, linewidth=0.9)
    for name, lon, lat, dx in [("Minneapolis", -93.2650, 44.9778, -1), ("St. Paul", -93.0900, 44.9537, 1)]:
        p = gpd.GeoSeries.from_xy([lon], [lat], crs=4326).to_crs(26915).iloc[0]
        ax.plot(p.x, p.y, "o", ms=2.6, color="white", mec=INK, mew=0.7)
        ax.annotate(name, (p.x, p.y), xytext=(dx * 5, 7), textcoords="offset points",
                    ha="right" if dx < 0 else "left", fontsize=6.8, fontweight="bold", color=INK)
    ax.set_axis_off()
    ax.set_aspect("equal")
    handles = [Patch(facecolor=c, label=l) for c, l in zip(RAMP, BIN_LABELS)]
    handles.append(Line2D([], [], color=INK, lw=0.9, label="METRO light rail"))
    ax.legend(handles=handles, loc="upper left", fontsize=6, frameon=False, handlelength=1.2,
              title="Jobs within 45 min, transit\n(avg. over 7–9 AM departures)",
              title_fontsize=6.2, alignment="left", borderaxespad=0.1)


def draw_chart(ax, s):
    c = pd.read_csv(OUT / "curve.csv")
    ax.plot(c["minutes"], c["transit"] / 1000, color=TRANSIT, lw=2)
    ax.plot(c["minutes"], c["walk"] / 1000, color=WALK, lw=2)
    ts = sorted(AO_2024)
    ax.plot(ts, [AO_2024[t] / 1000 for t in ts], "o", ms=4.2, mfc="white", mec=INK, mew=1, zorder=5)
    for t in (30, 45):
        ax.axvline(t, color=RULE, lw=0.8, zorder=0)
        tr, wk = s["thresholds"][str(t)]["transit_ww"], s["thresholds"][str(t)]["walk_ww"]
        ax.annotate(f"{k(tr)}", (t, tr / 1000), xytext=(-4, 6), textcoords="offset points",
                    ha="right", fontsize=6.5, color=INK, fontweight="bold")
        ax.annotate(f"{k(wk)}", (t, wk / 1000), xytext=(3, 4), textcoords="offset points",
                    ha="left", fontsize=6.5, color=INK2)
    ax.text(60.8, c["transit"].iloc[-1] / 1000, "Transit + walk", color=INK, fontsize=6.5, va="center")
    ax.text(60.8, c["walk"].iloc[-1] / 1000, "Walk only", color=INK, fontsize=6.5, va="center")
    ax.plot([], [], "o", ms=4, mfc="white", mec=INK, mew=1, label="AO published, Transit 2024")
    ax.legend(loc="upper left", fontsize=6.2, frameon=False, borderaxespad=0.2, handletextpad=0.3)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, None)
    ax.set_xticks([0, 10, 20, 30, 40, 45, 50, 60])
    ax.set_xlabel("Travel-time budget (minutes)", fontsize=6.8)
    ax.set_ylabel("Jobs reachable, avg. worker (thousands)", fontsize=6.8)
    ax.grid(axis="y", color=RULE, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.3, length=2)


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width))


def main():
    s = json.loads((OUT / "summary.json").read_text())
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    # ---- header ----
    fig.text(0.05, 0.962, "How many jobs can you reach by transit in the Twin Cities?",
             fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.05, 0.941,
             "A scaled-down replication of the Accessibility Observatory's Access Across America transit method, "
             "built to learn it. Not original research.", fontsize=8.2, color=INK2)
    fig.text(0.05, 0.925,
             f"Chinmay Arora  ·  7-county Metropolitan Council region  ·  {s['origin_bgs']:,} block groups  ·  "
             f"LODES {s['lodes_year']} jobs  ·  Metro Transit GTFS, Wed 7 Oct 2026, 7–9 AM",
             fontsize=6.8, color=MUTED)
    fig.add_artist(Line2D([0.05, 0.95], [0.915, 0.915], color=INK, lw=0.8))

    # ---- map ----
    ax_map = fig.add_axes([0.03, 0.455, 0.54, 0.455])
    draw_map(ax_map, s)
    fig.text(0.05, 0.905, "Map  ·  Jobs reachable within 45 minutes by transit, by block group",
             fontsize=7.4, fontweight="bold", color=INK)

    # ---- chart ----
    fig.text(0.60, 0.905, "Chart  ·  Jobs reachable by the average worker",
             fontsize=7.4, fontweight="bold", color=INK)
    ax_c = fig.add_axes([0.655, 0.695, 0.25, 0.195])
    draw_chart(ax_c, s)

    # ---- findings ----
    fig.text(0.60, 0.638, "Three findings", fontsize=9, fontweight="bold", color=INK)
    y = 0.621
    for i, (head, body) in enumerate(findings(s), 1):
        fig.text(0.60, y, f"{i}", fontsize=9, fontweight="bold", color=TRANSIT, va="top")
        fig.text(0.62, y, wrap(head, 60), fontsize=7.3, fontweight="bold", color=INK, va="top")
        txt = wrap(body, 58)
        fig.text(0.62, y - 0.0145, txt, fontsize=6.9, color=INK2, va="top", linespacing=1.35)
        y -= 0.0145 + 0.0118 * (txt.count("\n") + 1) + 0.012

    # ---- methods and limits box ----
    box_top, box_bot = 0.385, 0.068
    fig.add_artist(FancyBboxPatch((0.05, box_bot), 0.90, box_top - box_bot,
                                  boxstyle="round,pad=0,rounding_size=0.004",
                                  facecolor=PANEL, edgecolor=RULE, lw=0.6))
    fig.text(0.07, box_top - 0.022, "Methods and limits", fontsize=9, fontweight="bold", color=INK)
    method, limits = methods_and_limits(s)

    def column(x, title, rows, width):
        fig.text(x, box_top - 0.047, title, fontsize=7.3, fontweight="bold", color=TRANSIT)
        yy = box_top - 0.064
        for head, body in rows:
            txt = wrap(body, width)
            fig.text(x, yy, head, fontsize=6.8, fontweight="bold", color=INK, va="top")
            fig.text(x, yy - 0.0122, txt, fontsize=6.4, color=INK2, va="top", linespacing=1.3)
            yy -= 0.0122 + 0.0102 * (txt.count("\n") + 1) + 0.0065

    column(0.07, "What matches AO", method, 80)
    column(0.52, "Where this differs, and what it cannot say", limits, 80)

    # ---- footer ----
    fig.add_artist(Line2D([0.05, 0.95], [0.058, 0.058], color=RULE, lw=0.6))
    fig.text(0.05, 0.043, "Interactive map", fontsize=6.6, fontweight="bold", color=INK)
    short = lambda u: u.removeprefix("https://").rstrip("/")
    fig.text(0.16, 0.043, short(INTERACTIVE_URL), fontsize=6.6, color=TRANSIT, url=INTERACTIVE_URL)
    fig.text(0.05, 0.030, "Code + data", fontsize=6.6, fontweight="bold", color=INK)
    fig.text(0.16, 0.030, short(CODE_URL).replace("/tree/main", ""), fontsize=6.6, color=TRANSIT, url=CODE_URL)
    fig.text(0.95, 0.043, "Provenance tier: real public data, method replication", fontsize=6.6,
             fontweight="bold", color=INK, ha="right")
    fig.text(0.95, 0.030, "Method: Owen, Liu, Jain & Lind, Access Across America (UMN AO)", fontsize=6.3,
             color=MUTED, ha="right")

    fig.savefig(OUT / "twin_cities_transit_access_onepager.pdf")
    fig.savefig(OUT / "twin_cities_transit_access_onepager.png", dpi=150)
    print("wrote", OUT / "twin_cities_transit_access_onepager.pdf")


if __name__ == "__main__":
    main()
