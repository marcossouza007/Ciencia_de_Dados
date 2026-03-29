"""
Exploratory Data Analysis (EDA)
================================
Functions for descriptive statistics, distribution plots, correlation
analysis, and automated EDA reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Use a non-interactive backend so the module can run in headless environments
plt.switch_backend("Agg")


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return extended descriptive statistics including skewness and kurtosis."""
    stats = df.describe(include="all").T
    num_df = df.select_dtypes(include="number")
    stats["skewness"] = num_df.skew()
    stats["kurtosis"] = num_df.kurtosis()
    return stats


def value_counts_all(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Return value counts for each categorical column (top *top_n* values)."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return {col: df[col].value_counts().head(top_n) for col in cat_cols}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Plot histograms + KDE for each numeric column.

    If *output_dir* is provided, figures are saved as PNG files.
    """
    num_cols = columns or df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        if output_dir:
            path = Path(output_dir) / f"dist_{col}.png"
            fig.savefig(path, dpi=120)
            logger.info("Saved %s", path)
        plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    method: str = "pearson",
) -> None:
    """
    Generate and optionally save a correlation heatmap.

    Parameters
    ----------
    method : {'pearson', 'spearman', 'kendall'}
    """
    corr = df.select_dtypes(include="number").corr(method=method)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Correlation Heatmap ({method})")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120)
        logger.info("Saved correlation heatmap to %s", output_path)
    plt.close(fig)


def plot_boxplots(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Plot box plots for numeric columns to visualise spread and outliers."""
    num_cols = columns or df.select_dtypes(include="number").columns.tolist()
    fig, axes = plt.subplots(
        nrows=1, ncols=len(num_cols), figsize=(5 * len(num_cols), 5)
    )
    if len(num_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols):
        sns.boxplot(y=df[col].dropna(), ax=ax, color="lightcoral")
        ax.set_title(col)
    fig.tight_layout()
    if output_dir:
        path = Path(output_dir) / "boxplots.png"
        fig.savefig(path, dpi=120)
        logger.info("Saved boxplots to %s", path)
    plt.close(fig)


def plot_target_distribution(
    df: pd.DataFrame,
    target_column: str,
    output_path: Optional[str] = None,
) -> None:
    """Bar chart of target column class distribution."""
    counts = df[target_column].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color="teal", edgecolor="black")
    ax.set_title(f"Target Distribution: {target_column}")
    ax.set_xlabel(target_column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120)
        logger.info("Saved target distribution to %s", output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Automated EDA report
# ---------------------------------------------------------------------------

def generate_eda_report(
    df: pd.DataFrame,
    output_dir: str = "reports",
    target_column: Optional[str] = None,
) -> None:
    """
    Run the full EDA suite and save all artefacts to *output_dir*.

    Generates:
    - descriptive_stats.csv
    - correlation_heatmap.png
    - distribution_<col>.png for each numeric column
    - boxplots.png
    - target_distribution.png (if *target_column* is provided)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats = describe_dataframe(df)
    stats.to_csv(out / "descriptive_stats.csv")
    logger.info("Descriptive stats saved.")

    plot_distributions(df, output_dir=str(out))
    plot_correlation_heatmap(df, output_path=str(out / "correlation_heatmap.png"))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        plot_boxplots(df, columns=num_cols, output_dir=str(out))

    if target_column and target_column in df.columns:
        plot_target_distribution(
            df, target_column, output_path=str(out / "target_distribution.png")
        )

    logger.info("EDA report generated in '%s'.", out)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(7)
    df_demo = pd.DataFrame(
        {
            "age": rng.integers(18, 80, 500),
            "income": rng.exponential(55_000, 500),
            "score": rng.normal(0.6, 0.15, 500),
            "churn": rng.integers(0, 2, 500),
        }
    )

    print(describe_dataframe(df_demo))
    generate_eda_report(df_demo, output_dir="/tmp/eda_report", target_column="churn")
    print("EDA report saved to /tmp/eda_report")
