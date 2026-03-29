"""
Data Cleaning
=============
Utilities for detecting and handling missing values, duplicates, and outliers
in raw DataFrames before further analysis or modelling.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with missing-value counts and percentages."""
    total = df.isnull().sum()
    pct = total / len(df) * 100
    return (
        pd.DataFrame({"missing_count": total, "missing_pct": pct})
        .loc[total > 0]
        .sort_values("missing_pct", ascending=False)
    )


def drop_high_missing_columns(
    df: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """Drop columns whose missing-value ratio exceeds *threshold* (0–1)."""
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if cols_to_drop:
        logger.info("Dropping %d high-missing columns: %s", len(cols_to_drop), cols_to_drop)
    return df.drop(columns=cols_to_drop)


def impute_numeric(
    df: pd.DataFrame,
    strategy: str = "median",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Impute missing values in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : {'median', 'mean', 'zero'}
    columns : list of column names to impute; defaults to all numeric columns.
    """
    df = df.copy()
    num_cols = columns or df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            if strategy == "median":
                fill_value = df[col].median()
            elif strategy == "mean":
                fill_value = df[col].mean()
            elif strategy == "zero":
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy '{strategy}'")
            df[col] = df[col].fillna(fill_value)
            logger.debug("Imputed '%s' with %s=%.4f", col, strategy, fill_value)
    return df


def impute_categorical(
    df: pd.DataFrame,
    fill_value: str = "Unknown",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fill missing values in categorical/object columns with *fill_value*."""
    df = df.copy()
    cat_cols = columns or df.select_dtypes(include=["str", "category"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(fill_value)
    return df


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicate rows; returns a copy with duplicates dropped."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows.", removed)
    return df


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def clip_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Clip numeric outliers to [Q1 - factor*IQR, Q3 + factor*IQR].

    Returns a copy with outlier values clipped.
    """
    df = df.copy()
    num_cols = columns or df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def flag_outliers_zscore(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Add boolean flag columns (*_is_outlier) for numeric columns where
    the absolute Z-score exceeds *threshold*.
    """
    df = df.copy()
    num_cols = columns or df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        df[f"{col}_is_outlier"] = z.abs() > threshold
    return df


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def cast_dtypes(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    """Cast columns to the specified dtypes.

    Parameters
    ----------
    dtype_map : dict mapping column names to dtype strings, e.g.
        ``{"age": "int32", "price": "float32"}``.
    """
    df = df.copy()
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opinionated end-to-end cleaning pipeline:
    1. Drop columns with >50 % missing
    2. Remove duplicates
    3. Impute numeric columns with median
    4. Impute categorical columns with 'Unknown'
    5. Clip numeric outliers using 1.5×IQR rule
    """
    df = drop_high_missing_columns(df)
    df = remove_duplicates(df)
    df = impute_numeric(df, strategy="median")
    df = impute_categorical(df)
    df = clip_outliers_iqr(df)
    return df


# ---------------------------------------------------------------------------
# CLI entry-point (quick demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic demo dataset
    rng = np.random.default_rng(42)
    demo = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=200).astype(float),
            "income": rng.exponential(50_000, size=200),
            "category": rng.choice(["A", "B", "C", None], size=200),
            "score": rng.normal(0, 1, size=200),
        }
    )
    # Introduce some missings
    demo.loc[rng.choice(demo.index, 20, replace=False), "age"] = np.nan
    demo.loc[rng.choice(demo.index, 5, replace=False), :] = demo.iloc[0]  # duplicates

    print("Before cleaning:", demo.shape)
    print(report_missing(demo))
    cleaned = clean_pipeline(demo)
    print("After cleaning: ", cleaned.shape)
