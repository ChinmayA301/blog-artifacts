"""
AI Value Translation Audit
Author: Chinmay Arora

Run:
    python ai_value_translation_audit_analysis.py

Outputs:
    - ai_value_translation_audit_dataset.csv
    - category_summary.csv
    - audit_scores_by_use_case.png
    - category_average_scores.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

def load_data() -> pd.DataFrame:
    return pd.read_csv(OUT / "ai_value_translation_audit_dataset.csv")

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category")
        .agg(
            n=("use_case", "count"),
            avg_audit_score=("audit_score", "mean"),
            min_score=("audit_score", "min"),
            max_score=("audit_score", "max"),
        )
        .reset_index()
        .sort_values("avg_audit_score", ascending=False)
    )

def plot_use_case_scores(df: pd.DataFrame) -> None:
    ordered = df.sort_values("audit_score", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(ordered["use_case"], ordered["audit_score"])
    plt.xlabel("AI Value Translation Audit score / 10")
    plt.ylabel("Use case")
    plt.title("AI Value Translation Audit Scores by Use Case")
    plt.tight_layout()
    plt.savefig(OUT / "audit_scores_by_use_case.png", dpi=200)
    plt.close()

def plot_category_scores(summary_df: pd.DataFrame) -> None:
    ordered = summary_df.sort_values("avg_audit_score", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(ordered["category"], ordered["avg_audit_score"])
    plt.xlabel("Average audit score / 10")
    plt.ylabel("Category")
    plt.title("Average AI Value Translation Score by Category")
    plt.tight_layout()
    plt.savefig(OUT / "category_average_scores.png", dpi=200)
    plt.close()

def main() -> None:
    df = load_data()
    expected = df[["workflow_specificity", "data_readiness", "evaluation_loop", "accountability", "value_translation"]].sum(axis=1)
    if not (expected == df["audit_score"]).all():
        raise ValueError("Audit score mismatch.")

    summary_df = summarize(df)
    summary_df.to_csv(OUT / "category_summary.csv", index=False)

    plot_use_case_scores(df)
    plot_category_scores(summary_df)

    print("AI Value Translation Audit")
    print("=" * 32)
    print(df[["use_case", "audit_score", "category"]].sort_values("audit_score", ascending=False).to_string(index=False))
    print("\nCategory summary:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
