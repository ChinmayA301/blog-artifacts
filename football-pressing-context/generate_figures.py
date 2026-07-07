"""
Football pressing context study.

Builds an exploratory public-data sample from StatsBomb open data and produces
figures/models for the claim that raw forward pressure counts need tactical
context before they can explain player value or team outcomes.
"""

from __future__ import annotations

import json
import math
import os
import time
import unicodedata
import urllib.request
from urllib.error import HTTPError
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "savefig.dpi": 150,
    }
)

OUT = ROOT / "figures"
DATA = ROOT / "data"
CACHE = ROOT / ".cache" / "statsbomb-open-data"
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

INK = "#1A1A1A"
MUTE = "#9AA0A6"
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B3"


COMPETITIONS = [
    (11, 27, "La Liga", "2015/2016"),
    (9, 27, "1. Bundesliga", "2015/2016"),
    (9, 281, "1. Bundesliga", "2023/2024"),
    (7, 108, "Ligue 1", "2021/2022"),
    (7, 235, "Ligue 1", "2022/2023"),
    (43, 3, "FIFA World Cup", "2018"),
    (43, 106, "FIFA World Cup", "2022"),
    (55, 43, "UEFA Euro", "2020"),
    (55, 282, "UEFA Euro", "2024"),
    (16, 27, "Champions League", "2015/2016"),
    (16, 2, "Champions League", "2016/2017"),
    (16, 1, "Champions League", "2017/2018"),
    (16, 4, "Champions League", "2018/2019"),
]


TARGETS = {
    "Cristiano Ronaldo": {
        "aliases": ["Cristiano Ronaldo"],
        "role_group": "volume scorer",
    },
    "Lionel Messi": {
        "aliases": ["Lionel Messi", "Lionel Andres Messi"],
        "role_group": "creative forward",
    },
    "Karim Benzema": {
        "aliases": ["Karim Benzema"],
        "role_group": "connective striker",
    },
    "Robert Lewandowski": {
        "aliases": ["Robert Lewandowski"],
        "role_group": "volume scorer",
    },
    "Harry Kane": {
        "aliases": ["Harry Kane"],
        "role_group": "connective striker",
    },
    "Erling Haaland": {
        "aliases": ["Erling Haaland", "Erling Braut Haaland"],
        "role_group": "box striker",
    },
    "Kylian Mbappe": {
        "aliases": ["Kylian Mbappe", "Kylian Mbappe Lottin"],
        "role_group": "transition forward",
    },
    "Luis Suarez": {
        "aliases": ["Luis Suarez", "Luis Alberto Suarez"],
        "role_group": "volume scorer",
    },
    "Romelu Lukaku": {
        "aliases": ["Romelu Lukaku"],
        "role_group": "box striker",
    },
    "Olivier Giroud": {
        "aliases": ["Olivier Giroud"],
        "role_group": "box striker",
    },
    "Roberto Firmino": {
        "aliases": ["Roberto Firmino"],
        "role_group": "pressing forward",
    },
    "Gabriel Jesus": {
        "aliases": ["Gabriel Jesus", "Gabriel Fernando de Jesus"],
        "role_group": "pressing forward",
    },
}

ALIAS_LOOKUP = []
for canonical, meta in TARGETS.items():
    for alias in meta["aliases"]:
        ALIAS_LOOKUP.append((canonical, alias))


def norm(text: str) -> str:
    clean = unicodedata.normalize("NFKD", text)
    clean = "".join(ch for ch in clean if not unicodedata.combining(ch))
    return clean.lower()


def canonical_player(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = norm(name)
    for canonical, alias in ALIAS_LOOKUP:
        if norm(alias) in cleaned:
            return canonical
    return None


def fetch_json(path: str):
    cache_path = CACHE / path
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": "blog-artifacts-pressing-study"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read().decode("utf-8")
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            wait = 30 * (attempt + 1)
            print(f"rate limited fetching {path}; sleeping {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Could not fetch {path}")
    time.sleep(0.08)
    cache_path.write_text(payload)
    return json.loads(payload)


def clock_to_minutes(value: str | None, default: float = 90.0) -> float:
    if not value:
        return default
    value = str(value).split(".")[0]
    if ":" not in value:
        return float(value)
    minutes, seconds = value.split(":", 1)
    return float(minutes) + float(seconds) / 60.0


def player_minutes(lineup_player: dict) -> float:
    total = 0.0
    for pos in lineup_player.get("positions", []):
        start = clock_to_minutes(pos.get("from"), default=0.0)
        end = clock_to_minutes(pos.get("to"), default=90.0)
        total += max(0.0, end - start)
    return min(total, 130.0)


def location(event: dict) -> tuple[float | None, float | None]:
    loc = event.get("location")
    if not loc or len(loc) < 2:
        return None, None
    return float(loc[0]), float(loc[1])


def in_box(event: dict) -> bool:
    x, y = location(event)
    return x is not None and x >= 102 and 18 <= y <= 62


def progressive_reception(event: dict, target: str) -> bool:
    if event.get("type", {}).get("name") != "Pass":
        return False
    recipient = canonical_player(event.get("pass", {}).get("recipient", {}).get("name"))
    if recipient != target:
        return False
    x, _ = location(event)
    end = event.get("pass", {}).get("end_location") or []
    if x is None or len(end) < 2:
        return False
    end_x = float(end[0])
    return (end_x - x) >= 10 or (x < 80 <= end_x)


def empty_player_counts() -> dict:
    return {
        "pressures": 0,
        "counterpressures": 0,
        "attacking_third_pressures": 0,
        "middle_third_pressures": 0,
        "defensive_third_pressures": 0,
        "goals": 0,
        "shots": 0,
        "npxg": 0.0,
        "touches_box": 0,
        "progressive_receptions": 0,
    }


def empty_team_counts() -> dict:
    return {
        "passes": 0,
        "opp_buildout_passes_allowed": 0,
        "high_def_actions": 0,
        "xg": 0.0,
        "pressures": 0,
    }


def score_for_team(match: dict, team: str) -> tuple[int, int, int]:
    home = match["home_team"]["home_team_name"]
    away = match["away_team"]["away_team_name"]
    home_score = int(match["home_score"])
    away_score = int(match["away_score"])
    if team == home:
        goals_for, goals_against = home_score, away_score
    else:
        goals_for, goals_against = away_score, home_score
    points = 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
    return goals_for, goals_against, points


def build_match_rows(match: dict, competition: str, season: str) -> list[dict]:
    match_id = match["match_id"]
    lineups = fetch_json(f"lineups/{match_id}.json")
    player_team_minutes = {}

    for team_block in lineups:
        team = team_block["team_name"]
        for player in team_block.get("lineup", []):
            canonical = canonical_player(player.get("player_name"))
            if not canonical:
                continue
            mins = player_minutes(player)
            if mins > 0:
                player_team_minutes[(canonical, team)] = mins

    if not player_team_minutes:
        return []

    events = fetch_json(f"events/{match_id}.json")
    teams = [
        match["home_team"]["home_team_name"],
        match["away_team"]["away_team_name"],
    ]
    player_counts = defaultdict(empty_player_counts)
    team_counts = {team: empty_team_counts() for team in teams}

    for event in events:
        event_type = event.get("type", {}).get("name")
        team = event.get("team", {}).get("name")
        if team not in team_counts:
            continue
        opponent = teams[1] if team == teams[0] else teams[0]
        x, _ = location(event)

        if event_type == "Pass":
            team_counts[team]["passes"] += 1
            if x is not None and x <= 72:
                team_counts[opponent]["opp_buildout_passes_allowed"] += 1

        if event_type in {
            "Pressure",
            "Duel",
            "Interception",
            "Block",
            "Foul Committed",
            "Ball Recovery",
        }:
            if x is not None and x >= 48:
                team_counts[team]["high_def_actions"] += 1

        if event_type == "Pressure":
            team_counts[team]["pressures"] += 1

        if event_type == "Shot":
            shot = event.get("shot", {})
            xg = float(shot.get("statsbomb_xg") or 0.0)
            team_counts[team]["xg"] += xg

        player = canonical_player(event.get("player", {}).get("name"))
        if not player:
            for target in TARGETS:
                if progressive_reception(event, target):
                    player_counts[(target, team)]["progressive_receptions"] += 1
            continue

        counts = player_counts[(player, team)]

        if event_type == "Pressure":
            counts["pressures"] += 1
            if event.get("counterpress"):
                counts["counterpressures"] += 1
            if x is not None:
                if x >= 80:
                    counts["attacking_third_pressures"] += 1
                elif x >= 40:
                    counts["middle_third_pressures"] += 1
                else:
                    counts["defensive_third_pressures"] += 1

        if event_type == "Shot":
            shot = event.get("shot", {})
            counts["shots"] += 1
            if shot.get("outcome", {}).get("name") == "Goal":
                counts["goals"] += 1
            if shot.get("type", {}).get("name") != "Penalty":
                counts["npxg"] += float(shot.get("statsbomb_xg") or 0.0)

        if event_type in {
            "Ball Receipt*",
            "Carry",
            "Dribble",
            "Pass",
            "Shot",
            "Miscontrol",
            "Dispossessed",
        } and in_box(event):
            counts["touches_box"] += 1

        if progressive_reception(event, player):
            counts["progressive_receptions"] += 1

    rows = []
    for (player, team), minutes in player_team_minutes.items():
        goals_for, goals_against, points = score_for_team(match, team)
        opponent = teams[1] if team == teams[0] else teams[0]
        t = team_counts[team]
        opp = team_counts[opponent]
        ppda_proxy = t["opp_buildout_passes_allowed"] / max(t["high_def_actions"], 1)
        possession_proxy = t["passes"] / max(t["passes"] + opp["passes"], 1)
        counts = player_counts[(player, team)]

        rows.append(
            {
                "player": player,
                "team": team,
                "competition": competition,
                "season": season,
                "match_id": match_id,
                "match_date": match.get("match_date"),
                "minutes": minutes,
                "role_group": TARGETS[player]["role_group"],
                "team_goals_for": goals_for,
                "team_goals_against": goals_against,
                "team_points": points,
                "team_xg": t["xg"],
                "opponent_xg": opp["xg"],
                "xg_difference": t["xg"] - opp["xg"],
                "team_ppda_proxy": ppda_proxy,
                "possession_proxy": possession_proxy,
                "team_pressures": t["pressures"],
                **counts,
            }
        )

    return rows


def build_dataset(min_minutes: int = 90) -> pd.DataFrame:
    match_rows = []
    for competition_id, season_id, competition, season in COMPETITIONS:
        print(f"loading {competition} {season}")
        matches = fetch_json(f"matches/{competition_id}/{season_id}.json")
        for match in matches:
            match_rows.extend(build_match_rows(match, competition, season))

    match_df = pd.DataFrame(match_rows)
    if match_df.empty:
        raise RuntimeError("No target-player rows found in the selected open-data scope.")

    weighted_cols = [
        "team_points",
        "team_xg",
        "opponent_xg",
        "xg_difference",
        "team_ppda_proxy",
        "possession_proxy",
        "team_pressures",
    ]
    count_cols = [
        "pressures",
        "counterpressures",
        "attacking_third_pressures",
        "middle_third_pressures",
        "defensive_third_pressures",
        "goals",
        "shots",
        "npxg",
        "touches_box",
        "progressive_receptions",
    ]

    records = []
    keys = ["player", "team", "competition", "season", "role_group"]
    for key, g in match_df.groupby(keys, dropna=False):
        total_minutes = g["minutes"].sum()
        if total_minutes < min_minutes:
            continue
        row = dict(zip(keys, key))
        row["minutes"] = total_minutes
        row["matches"] = g["match_id"].nunique()
        for col in count_cols:
            row[col] = g[col].sum()
            row[f"{col}_p90"] = row[col] / total_minutes * 90
        for col in weighted_cols:
            row[col] = np.average(g[col], weights=g["minutes"])
        row["points_per_match"] = row["team_points"]
        row["team_outcome_tier"] = (
            "high"
            if row["points_per_match"] >= 2.0
            else "middle"
            if row["points_per_match"] >= 1.25
            else "low"
        )
        records.append(row)

    df = pd.DataFrame(records).sort_values(["player", "season", "competition"])
    df.to_csv(DATA / "forward_pressing_context_dataset.csv", index=False)
    print(f"dataset rows: {len(df)} | written to data/forward_pressing_context_dataset.csv")
    return df


def fit_models(df: pd.DataFrame) -> dict:
    model_df = df[
        [
            "points_per_match",
            "pressures_p90",
            "npxg_p90",
            "goals_p90",
            "touches_box_p90",
            "progressive_receptions_p90",
            "team_ppda_proxy",
            "possession_proxy",
            "xg_difference",
        ]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    models = {
        "m1_naive_pressing": smf.ols(
            "points_per_match ~ pressures_p90",
            data=model_df,
        ).fit(),
        "m2_attacking_tradeoff": smf.ols(
            """
            points_per_match ~ pressures_p90
                             + npxg_p90
                             + goals_p90
                             + touches_box_p90
                             + progressive_receptions_p90
            """,
            data=model_df,
        ).fit(),
        "m3_team_context": smf.ols(
            """
            points_per_match ~ pressures_p90
                             + team_ppda_proxy
                             + possession_proxy
                             + xg_difference
                             + npxg_p90
                             + goals_p90
            """,
            data=model_df,
        ).fit(),
        "m4_interaction": smf.ols(
            """
            points_per_match ~ pressures_p90 * team_ppda_proxy
                             + possession_proxy
                             + xg_difference
                             + npxg_p90
                             + goals_p90
            """,
            data=model_df,
        ).fit(),
    }

    summary_path = ROOT / "model_summaries.txt"
    with summary_path.open("w") as handle:
        for name, model in models.items():
            handle.write(f"\n\n{name}\n")
            handle.write("=" * len(name) + "\n")
            handle.write(str(model.summary()))
            handle.write("\n")
    print(f"model summaries written to {summary_path.name}")
    return models


def label_key_points(ax, df: pd.DataFrame, x_col: str, y_col: str, max_labels: int = 18):
    key_names = {
        "Cristiano Ronaldo",
        "Lionel Messi",
        "Karim Benzema",
        "Robert Lewandowski",
        "Harry Kane",
        "Erling Haaland",
        "Kylian Mbappe",
        "Roberto Firmino",
    }
    labels = df[df["player"].isin(key_names)].sort_values("minutes", ascending=False).head(max_labels)
    for _, row in labels.iterrows():
        ax.annotate(
            row["player"].split()[-1],
            (row[x_col], row[y_col]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )


def figure_naive(df: pd.DataFrame):
    role_colors = {
        "volume scorer": BLUE,
        "creative forward": PURPLE,
        "connective striker": GREEN,
        "box striker": ORANGE,
        "transition forward": RED,
        "pressing forward": INK,
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for role, g in df.groupby("role_group"):
        ax.scatter(
            g["pressures_p90"],
            g["points_per_match"],
            s=np.clip(g["goals_p90"] * 180 + 45, 45, 260),
            alpha=0.78,
            color=role_colors.get(role, MUTE),
            label=role,
            edgecolor="white",
            linewidth=0.7,
        )
    label_key_points(ax, df, "pressures_p90", "points_per_match")
    ax.set_xlabel("Forward pressure events per 90")
    ax.set_ylabel("Team points per match in sample")
    ax.set_title("Raw forward pressing does not cleanly explain team outcomes", loc="left")
    ax.legend(loc="best", frameon=True, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0,
        -0.17,
        "Bubble size is goals per 90. The open-data sample mixes full and partial competition coverage, so this is exploratory, not a league-wide ranking.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTE,
        va="top",
    )
    plt.tight_layout()
    plt.savefig(OUT / "pressing_naive_relationship.png", bbox_inches="tight")
    plt.close()


def figure_tradeoff(df: pd.DataFrame):
    tier_colors = {"high": GREEN, "middle": ORANGE, "low": RED}
    plot_df = df.copy()
    plot_df["attacking_tradeoff"] = plot_df["goals_p90"] + plot_df["npxg_p90"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for tier, g in plot_df.groupby("team_outcome_tier"):
        ax.scatter(
            g["pressures_p90"],
            g["attacking_tradeoff"],
            s=np.clip(g["minutes"] / 8, 40, 240),
            alpha=0.78,
            color=tier_colors[tier],
            label=f"{tier} outcome tier",
            edgecolor="white",
            linewidth=0.7,
        )
    label_key_points(ax, plot_df, "pressures_p90", "attacking_tradeoff")
    ax.set_xlabel("Forward pressure events per 90")
    ax.set_ylabel("Goals + non-penalty xG per 90")
    ax.set_title("Pressing volume must be weighed against attacking tradeoff", loc="left")
    ax.legend(loc="best", frameon=True, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0,
        -0.17,
        "High-value attackers can sit in low-pressure regions without automatically implying low team value; the question is whether the system can afford the cost.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTE,
        va="top",
    )
    plt.tight_layout()
    plt.savefig(OUT / "pressing_attacking_tradeoff.png", bbox_inches="tight")
    plt.close()


def figure_interaction(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    scatter = ax.scatter(
        df["team_ppda_proxy"],
        df["pressures_p90"],
        c=df["xg_difference"],
        cmap="RdYlGn",
        s=np.clip(df["minutes"] / 8, 45, 240),
        alpha=0.82,
        edgecolor="white",
        linewidth=0.7,
    )
    label_key_points(ax, df, "team_ppda_proxy", "pressures_p90", max_labels=20)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Team xG difference per match")
    ax.set_xlabel("Team PPDA-style proxy (lower = more pressure context)")
    ax.set_ylabel("Forward pressure events per 90")
    ax.set_title("Individual pressure volume is a team-style interaction", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0,
        -0.17,
        "The PPDA proxy uses opponent buildout passes allowed divided by high defensive actions from StatsBomb events. It is a public-data approximation, not an Opta PPDA clone.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTE,
        va="top",
    )
    plt.tight_layout()
    plt.savefig(OUT / "team_style_interaction.png", bbox_inches="tight")
    plt.close()


def figure_ronaldo(df: pd.DataFrame):
    r = df[df["player"] == "Cristiano Ronaldo"].copy()
    if r.empty:
        return
    r["label"] = r["competition"] + "\n" + r["season"] + "\n" + r["team"]
    r = r.sort_values(["season", "competition", "team"])
    x = np.arange(len(r))

    fig, ax1 = plt.subplots(figsize=(11.5, 6.5))
    ax1.bar(x, r["pressures_p90"], color=BLUE, alpha=0.82, label="Pressures per 90")
    ax1.set_ylabel("Pressure events per 90")
    ax1.set_xticks(x)
    ax1.set_xticklabels(r["label"], rotation=35, ha="right", fontsize=8.5)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        r["goals_p90"] + r["npxg_p90"],
        color=RED,
        marker="o",
        lw=2.2,
        label="Goals + npxG per 90",
    )
    ax2.set_ylabel("Goals + non-penalty xG per 90")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True, fontsize=8.5)
    ax1.set_title("Ronaldo's low-pressing question is a recurring tradeoff", loc="left")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.text(
        0,
        -0.28,
        "StatsBomb open-data coverage for Ronaldo is mostly international tournaments, Clasico samples, and Champions League finals. Use this as a public-data anchor, not a full club-season timeline.",
        transform=ax1.transAxes,
        fontsize=8.5,
        color=MUTE,
        va="top",
    )
    plt.tight_layout()
    plt.savefig(OUT / "ronaldo_timeline.png", bbox_inches="tight")
    plt.close()


def figure_context_ladder():
    steps = [
        "Raw pressure count",
        "Zone-adjusted pressure",
        "Game-state-adjusted pressure",
        "Team-style-adjusted pressure",
        "Role-adjusted pressure",
        "Outcome-adjusted pressure value",
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")
    y_positions = np.linspace(0.86, 0.12, len(steps))
    for idx, (step, y) in enumerate(zip(steps, y_positions)):
        color = BLUE if idx == 0 else GREEN if idx == len(steps) - 1 else MUTE
        ax.text(
            0.5,
            y,
            step,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold" if idx in {0, len(steps) - 1} else "normal",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=color, lw=1.8),
        )
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[idx + 1] + 0.055),
                xytext=(0.5, y - 0.055),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4),
            )
    ax.set_title("The context ladder for pressing value", loc="left", pad=18)
    ax.text(
        0.02,
        0.01,
        "A pressure event without distance, angle, speed, teammate synchronization, role, and game state is an incomplete observation.",
        transform=ax.transAxes,
        fontsize=9,
        color=MUTE,
        va="bottom",
    )
    plt.tight_layout()
    plt.savefig(OUT / "context_ladder.png", bbox_inches="tight")
    plt.close()


def write_run_notes(models: dict, df: pd.DataFrame):
    lines = [
        "Run notes",
        "---------",
        f"Rows in derived sample: {len(df)}",
        f"Players: {df['player'].nunique()}",
        f"Player-team-competition-season samples: {len(df)}",
        "",
    ]
    for name, model in models.items():
        pressure_coef = model.params.get("pressures_p90", math.nan)
        pressure_p = model.pvalues.get("pressures_p90", math.nan)
        lines.append(
            f"{name}: R2={model.rsquared:.3f}, pressure coef={pressure_coef:.4f}, p={pressure_p:.3f}"
        )
    (ROOT / "run_notes.txt").write_text("\n".join(lines) + "\n")


def main():
    min_minutes = int(os.environ.get("MIN_MINUTES", "90"))
    df = build_dataset(min_minutes=min_minutes)
    models = fit_models(df)
    figure_naive(df)
    figure_tradeoff(df)
    figure_interaction(df)
    figure_ronaldo(df)
    figure_context_ladder()
    write_run_notes(models, df)
    print("figures done")


if __name__ == "__main__":
    main()
