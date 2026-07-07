"""
Physical-output context study for World Cup discourse.

Purpose:
- Reproduce the public aggregate comparison used in the blog.
- Show why a claim like "Ronaldo does not move" is too shallow.
- Keep the caveat explicit: this is aggregate reported data, not raw FIFA tracking data.

Inputs:
- No external file required by default.
- Optional: replace the hard-coded rows with a CSV containing:
  player,avg_speed_kmh,high_speed_runs,sprints,distance_m

Outputs:
- physical_output_context_table.csv
- physical_output_index_bar.png
- physical_output_relative_comparison.csv
"""

from __future__ import annotations

from pathlib import Path
import os

OUT_DIR = Path(__file__).resolve().parent / "outputs"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".matplotlib"))

import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR.mkdir(exist_ok=True)


def build_default_dataset() -> pd.DataFrame:
    """Public aggregate physical-output figures cited in the blog draft."""
    return pd.DataFrame(
        [
            {"player": "Kylian Mbappe", "avg_speed_kmh": 5.43, "high_speed_runs": 208, "sprints": 109, "distance_m": 26305.88},
            {"player": "Cristiano Ronaldo", "avg_speed_kmh": 5.04, "high_speed_runs": 197, "sprints": 97, "distance_m": 25274.27},
            {"player": "Erling Haaland", "avg_speed_kmh": 3.82, "high_speed_runs": 160, "sprints": 81, "distance_m": 19816.22},
            {"player": "Lionel Messi", "avg_speed_kmh": 4.58, "high_speed_runs": 151, "sprints": 69, "distance_m": 17230.39},
        ]
    )


def add_physical_output_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight min-max index across speed, high-speed runs, sprints, and distance.
    This is intentionally simple and descriptive, not a player-quality model.
    """
    df = df.copy()
    metrics = ["avg_speed_kmh", "high_speed_runs", "sprints", "distance_m"]

    for metric in metrics:
        denom = df[metric].max() - df[metric].min()
        if denom == 0:
            df[f"{metric}_scaled"] = 0.0
        else:
            df[f"{metric}_scaled"] = (df[metric] - df[metric].min()) / denom

    scaled_cols = [f"{metric}_scaled" for metric in metrics]
    df["physical_output_index"] = df[scaled_cols].mean(axis=1) * 100
    return df.sort_values("physical_output_index", ascending=False)


def relative_comparison(df: pd.DataFrame, target: str = "Cristiano Ronaldo") -> pd.DataFrame:
    """Compare target player to everyone else by percentage difference."""
    df = df.copy()
    target_row = df.loc[df["player"].eq(target)]
    if target_row.empty:
        raise ValueError(f"Target player not found: {target}")
    target_row = target_row.iloc[0]

    rows = []
    for _, row in df.iterrows():
        if row["player"] == target:
            continue
        rows.append(
            {
                "comparison": f"{target} vs {row['player']}",
                "distance_pct_diff": (target_row["distance_m"] / row["distance_m"] - 1) * 100,
                "high_speed_runs_pct_diff": (target_row["high_speed_runs"] / row["high_speed_runs"] - 1) * 100,
                "sprints_pct_diff": (target_row["sprints"] / row["sprints"] - 1) * 100,
                "avg_speed_pct_diff": (target_row["avg_speed_kmh"] / row["avg_speed_kmh"] - 1) * 100,
            }
        )
    return pd.DataFrame(rows)


def plot_index(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df["player"], df["physical_output_index"])
    plt.ylabel("Physical Output Index (0-100)")
    plt.title("Aggregate Physical Output Index: Selected World Cup Forwards")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "physical_output_index_bar.png", dpi=200)
    plt.close()


def main() -> None:
    df = build_default_dataset()
    df = add_physical_output_index(df)
    rel = relative_comparison(df)

    df.to_csv(OUT_DIR / "physical_output_context_table.csv", index=False)
    rel.to_csv(OUT_DIR / "physical_output_relative_comparison.csv", index=False)
    plot_index(df)

    print("Physical output table")
    print(df[["player", "avg_speed_kmh", "high_speed_runs", "sprints", "distance_m", "physical_output_index"]].round(2))
    print("\nRelative comparison")
    print(rel.round(2))
    print(f"\nOutputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
