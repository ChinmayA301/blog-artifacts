"""
Screenshot trap study template.

Purpose:
- Turn viral screenshot claims into repeated-event coding.
- Test whether a selected frame is representative of a repeated pattern.

Workflow:
1. Pick a match and player.
2. Collect criticized screenshots/clips.
3. Identify comparable situations in the same match or tournament.
4. Manually code each situation using this schema.
5. Compare bad-frame rate, correct-action rate, invisible-positive-action rate, and context.

This is designed for manual or semi-manual coding. It is intentionally transparent.
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

SCHEMA = [
    "match_id",
    "minute",
    "second",
    "player",
    "team",
    "opponent",
    "score_state",  # winning/drawing/losing + score
    "phase",  # buildup, settled_defense, press, transition_attack, transition_defense, box_attack
    "situation_type",  # press_trigger, channel_run, box_position, decoy_run, turnover, etc.
    "screenshot_claim",  # e.g. lazy_press, missed_run, standing_still, poor_position
    "comparable_event_id",  # group similar moments together
    "role_instruction_hypothesis",  # hold_lane, press_cb, conserve_box, attack_near_post, unknown
    "action_grade",  # -2 bad, -1 weak, 0 neutral, 1 good, 2 excellent
    "team_state_change",  # -2 worsened a lot to +2 improved a lot
    "received_pass",  # 1/0
    "created_space",  # 1/0
    "blocked_lane",  # 1/0
    "team_synchronized",  # 1/0
    "shot_within_10s",  # 1/0
    "turnover_within_10s",  # 1/0
    "notes",
]


def create_blank_template(path: Path = OUT_DIR / "screenshot_trap_coding_template.csv") -> None:
    pd.DataFrame(columns=SCHEMA).to_csv(path, index=False)
    print(f"Blank coding template written to {path}")


def load_or_demo(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    # Demo rows show how to code; replace with your manually coded events.
    return pd.DataFrame(
        [
            {
                "match_id": "POR-ESP-2026-R16",
                "minute": 12,
                "second": 15,
                "player": "Cristiano Ronaldo",
                "team": "Portugal",
                "opponent": "Spain",
                "score_state": "0-0",
                "phase": "press",
                "situation_type": "press_trigger",
                "screenshot_claim": "standing_still",
                "comparable_event_id": "press_cb_group_1",
                "role_instruction_hypothesis": "hold_lane",
                "action_grade": 1,
                "team_state_change": 1,
                "received_pass": 0,
                "created_space": 0,
                "blocked_lane": 1,
                "team_synchronized": 1,
                "shot_within_10s": 0,
                "turnover_within_10s": 0,
                "notes": "Looks passive in still frame, but blocks central lane.",
            },
            {
                "match_id": "POR-ESP-2026-R16",
                "minute": 27,
                "second": 42,
                "player": "Cristiano Ronaldo",
                "team": "Portugal",
                "opponent": "Spain",
                "score_state": "0-0",
                "phase": "press",
                "situation_type": "press_trigger",
                "screenshot_claim": "missed_press",
                "comparable_event_id": "press_cb_group_1",
                "role_instruction_hypothesis": "press_cb",
                "action_grade": -1,
                "team_state_change": -1,
                "received_pass": 0,
                "created_space": 0,
                "blocked_lane": 0,
                "team_synchronized": 0,
                "shot_within_10s": 0,
                "turnover_within_10s": 0,
                "notes": "Poor trigger; team did not move with him.",
            },
        ]
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    required = {"player", "situation_type", "action_grade", "team_state_change", "created_space", "blocked_lane", "team_synchronized"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    summary = (
        df.groupby(["player", "situation_type"])
        .agg(
            moments=("action_grade", "count"),
            avg_action_grade=("action_grade", "mean"),
            avg_team_state_change=("team_state_change", "mean"),
            negative_rate=("action_grade", lambda x: (x < 0).mean()),
            positive_rate=("action_grade", lambda x: (x > 0).mean()),
            created_space_rate=("created_space", "mean"),
            blocked_lane_rate=("blocked_lane", "mean"),
            team_sync_rate=("team_synchronized", "mean"),
        )
        .reset_index()
    )
    return summary


def plot_negative_rate(summary: pd.DataFrame) -> None:
    labels = summary["player"] + " | " + summary["situation_type"]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, summary["negative_rate"])
    plt.ylabel("Negative Rate")
    plt.title("Screenshot Trap Check: How Often Was the Action Actually Negative?")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "screenshot_trap_negative_rate.png", dpi=200)
    plt.close()


def main() -> None:
    create_blank_template()
    coded_path = OUT_DIR / "screenshot_trap_coded_events.csv"
    df = load_or_demo(coded_path)
    df.to_csv(coded_path, index=False)

    summary = summarize(df)
    summary.to_csv(OUT_DIR / "screenshot_trap_summary.csv", index=False)
    plot_negative_rate(summary)

    print(summary.round(3))
    print(f"\nOutputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
