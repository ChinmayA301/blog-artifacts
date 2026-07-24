"""Analyze the ASER access/capability panel and enforcement denominators."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def bootstrap_pearson(
    x: np.ndarray, y: np.ndarray, iterations: int = 10_000
) -> tuple[float, float]:
    rng = np.random.default_rng(20260723)
    estimates: list[float] = []
    n = len(x)
    for _ in range(iterations):
        indices = rng.integers(0, n, n)
        sample_x = x[indices]
        sample_y = y[indices]
        if np.std(sample_x) == 0 or np.std(sample_y) == 0:
            continue
        estimates.append(float(np.corrcoef(sample_x, sample_y)[0, 1]))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def permutation_pvalue(
    x: np.ndarray, y: np.ndarray, iterations: int = 20_000
) -> float:
    rng = np.random.default_rng(20260723)
    observed = abs(float(np.corrcoef(x, y)[0, 1]))
    exceedances = 0
    for _ in range(iterations):
        permuted = rng.permutation(y)
        estimate = abs(float(np.corrcoef(x, permuted)[0, 1]))
        exceedances += estimate >= observed
    return (exceedances + 1) / (iterations + 1)


def metric_substitution_results(panel: pd.DataFrame) -> dict[str, object]:
    enrollment = panel["enrollment_age_6_14_pct"].to_numpy()
    reading = panel["std3_read_std2_text_pct"].to_numpy()
    pearson_r = float(np.corrcoef(enrollment, reading)[0, 1])
    spearman_rho = float(
        pd.Series(enrollment).rank().corr(pd.Series(reading).rank())
    )
    slope, intercept = np.polyfit(enrollment, reading, 1)
    ci_low, ci_high = bootstrap_pearson(enrollment, reading)
    p_value = permutation_pvalue(enrollment, reading)

    return {
        "n_states_uts": int(len(panel)),
        "unweighted_state_mean_enrollment_pct": float(enrollment.mean()),
        "unweighted_state_mean_reading_pct": float(reading.mean()),
        "enrollment_range_pct": [
            float(enrollment.min()),
            float(enrollment.max()),
        ],
        "reading_range_pct": [float(reading.min()), float(reading.max())],
        "enrollment_cv": float(enrollment.std(ddof=1) / enrollment.mean()),
        "reading_cv": float(reading.std(ddof=1) / reading.mean()),
        "pearson_r": pearson_r,
        "pearson_permutation_p": p_value,
        "pearson_bootstrap_95_ci": [ci_low, ci_high],
        "spearman_rho": spearman_rho,
        "ols_intercept": float(intercept),
        "ols_slope_reading_pp_per_enrollment_pp": float(slope),
        "ols_r_squared": pearson_r**2,
        "mean_access_capability_wedge_pp": float(
            panel["access_capability_wedge_pp"].mean()
        ),
        "widest_wedge_state": str(
            panel.loc[panel["access_capability_wedge_pp"].idxmax(), "state"]
        ),
        "narrowest_wedge_state": str(
            panel.loc[panel["access_capability_wedge_pp"].idxmin(), "state"]
        ),
    }


def enforcement_results(ed: pd.DataFrame) -> dict[str, float]:
    counts = dict(zip(ed["metric"], ed["count"]))
    convictions = float(counts["Trial cases resulting in conviction"])
    completed = float(counts["Trials completed"])
    complaints = float(counts["Prosecution complaints filed"])
    ecirs = float(counts["ECIR recorded"])
    return {
        "convictions_per_completed_trial_pct": convictions / completed * 100,
        "convictions_per_prosecution_complaint_pct": (
            convictions / complaints * 100
        ),
        "convictions_per_ecir_pct": convictions / ecirs * 100,
        "completed_trials_per_ecir_pct": completed / ecirs * 100,
        "complaints_per_ecir_pct": complaints / ecirs * 100,
    }


def plot_scatter(panel: pd.DataFrame, results: dict[str, object]) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        panel["enrollment_age_6_14_pct"],
        panel["std3_read_std2_text_pct"],
        c=panel["access_capability_wedge_pp"],
        cmap="magma_r",
        s=85,
        edgecolor="#10253f",
        linewidth=0.5,
    )

    for _, row in panel.iterrows():
        if (
            row["std3_read_std2_text_pct"]
            in (
                panel["std3_read_std2_text_pct"].min(),
                panel["std3_read_std2_text_pct"].max(),
            )
            or row["access_capability_wedge_pp"]
            == panel["access_capability_wedge_pp"].max()
        ):
            ax.annotate(
                row["state"],
                (
                    row["enrollment_age_6_14_pct"],
                    row["std3_read_std2_text_pct"],
                ),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_title("ASER 2024: Access is saturated; learning still varies")
    ax.set_xlabel("Children age 6-14 enrolled in school (%)")
    ax.set_ylabel("Std III children reading a Std II text (%)")
    ax.grid(alpha=0.18)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Access-capability wedge (percentage points)")
    ax.text(
        0.01,
        0.01,
        f"Pearson r = {results['pearson_r']:.2f}; "
        f"n = {results['n_states_uts']} states/UTs",
        transform=ax.transAxes,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "aser_access_vs_learning.png", dpi=220)
    plt.close(fig)


def plot_denominators(ed: pd.DataFrame, results: dict[str, float]) -> None:
    labels = ["Completed trials", "Prosecution complaints", "Recorded ECIRs"]
    values = [
        results["convictions_per_completed_trial_pct"],
        results["convictions_per_prosecution_complaint_pct"],
        results["convictions_per_ecir_pct"],
    ]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    bars = ax.barh(labels, values, color=["#d95f46", "#e6a44a", "#2b6f8e"])
    ax.invert_yaxis()
    ax.set_xlabel("56 conviction outcomes as a share of denominator (%)")
    ax.set_title("The denominator changes the question")
    ax.grid(axis="x", alpha=0.18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
        )
    ax.set_xlim(0, 102)
    fig.tight_layout()
    fig.savefig(FIGURES / "enforcement_denominator_lab.png", dpi=220)
    plt.close(fig)


def main() -> None:
    panel = pd.read_csv(DATA / "aser_2024_state_proxy_capability.csv")
    ed = pd.read_csv(DATA / "ed_denominator_snapshot.csv")

    metric_results = metric_substitution_results(panel)
    denominator_results = enforcement_results(ed)
    results = {
        "metric_substitution": metric_results,
        "enforcement_denominators": denominator_results,
    }

    with (ROOT / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    summary = panel.sort_values(
        "access_capability_wedge_pp", ascending=False
    )[
        [
            "state",
            "enrollment_age_6_14_pct",
            "std3_read_std2_text_pct",
            "access_capability_wedge_pp",
        ]
    ]
    summary.to_csv(ROOT / "state_wedge_summary.csv", index=False)

    plot_scatter(panel, metric_results)
    plot_denominators(ed, denominator_results)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
