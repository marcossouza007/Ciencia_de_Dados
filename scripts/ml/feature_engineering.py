"""
Feature Engineering
===================
Creates new features from raw columns: polynomial interactions, datetime
decomposition, binning, and target encoding.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interaction / polynomial features
# ---------------------------------------------------------------------------

def add_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2,
    include_bias: bool = False,
) -> pd.DataFrame:
    """
    Append polynomial and interaction features for the given columns.

    The original columns are kept; new columns are named automatically by
    sklearn's PolynomialFeatures.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_array = poly.fit_transform(df[columns])
    feature_names = poly.get_feature_names_out(columns)
    poly_df = pd.DataFrame(poly_array, columns=feature_names, index=df.index)
    # Drop the first-degree terms to avoid duplication
    new_cols = [c for c in feature_names if c not in columns]
    return pd.concat([df, poly_df[new_cols]], axis=1)


# ---------------------------------------------------------------------------
# Datetime features
# ---------------------------------------------------------------------------

def extract_datetime_features(
    df: pd.DataFrame,
    column: str,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Decompose a datetime column into year, month, day, day-of-week, hour,
    is_weekend, and a cyclical hour encoding.
    """
    df = df.copy()
    dt = pd.to_datetime(df[column])
    df[f"{column}_year"] = dt.dt.year
    df[f"{column}_month"] = dt.dt.month
    df[f"{column}_day"] = dt.dt.day
    df[f"{column}_dayofweek"] = dt.dt.dayofweek
    df[f"{column}_hour"] = dt.dt.hour
    df[f"{column}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    # Cyclical encoding for hour
    df[f"{column}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df[f"{column}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    if drop_original:
        df = df.drop(columns=[column])
    return df


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_column(
    df: pd.DataFrame,
    column: str,
    bins: int | list = 5,
    labels: Optional[List[str]] = None,
    new_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Bin a numeric column into discrete intervals.

    Parameters
    ----------
    bins : int or list of bin edges
    labels : optional list of labels for the bins
    new_column : name for the binned column; defaults to ``<column>_bin``.
    """
    df = df.copy()
    out_col = new_column or f"{column}_bin"
    df[out_col] = pd.cut(df[column], bins=bins, labels=labels)
    return df


# ---------------------------------------------------------------------------
# Ratio / log transforms
# ---------------------------------------------------------------------------

def add_ratio_feature(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    new_column: Optional[str] = None,
    fill_inf: float = 0.0,
) -> pd.DataFrame:
    """Add ``numerator / denominator`` as a new feature, guarding against division by zero."""
    df = df.copy()
    out_col = new_column or f"{numerator}_per_{denominator}"
    df[out_col] = df[numerator] / df[denominator].replace(0, np.nan)
    df[out_col] = df[out_col].replace([np.inf, -np.inf], fill_inf).fillna(fill_inf)
    return df


def log_transform(
    df: pd.DataFrame,
    columns: List[str],
    shift: float = 1.0,
) -> pd.DataFrame:
    """Apply log(x + shift) transform to the specified columns.

    Uses ``np.log1p`` when *shift* equals 1 (default), otherwise uses
    ``np.log``.  A *shift* > 0 ensures the argument is positive when values
    may be zero.
    """
    df = df.copy()
    for col in columns:
        df[col] = np.log(df[col] + shift)
    return df


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

def target_encode(
    df: pd.DataFrame,
    categorical_columns: List[str],
    target_column: str,
    smoothing: float = 10.0,
) -> pd.DataFrame:
    """
    Replace each category with a smoothed estimate of the target mean.

    ``encoded = (count * cat_mean + smoothing * global_mean) / (count + smoothing)``
    """
    df = df.copy()
    global_mean = df[target_column].mean()
    for col in categorical_columns:
        agg = df.groupby(col)[target_column].agg(["mean", "count"])
        agg["smoothed"] = (
            (agg["count"] * agg["mean"] + smoothing * global_mean)
            / (agg["count"] + smoothing)
        )
        df[col] = df[col].map(agg["smoothed"])
    return df


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, 200),
            "income": rng.exponential(40_000, 200),
            "city": rng.choice(["NY", "LA", "SF"], 200),
            "signup_date": pd.date_range("2022-01-01", periods=200, freq="3h"),
            "churn": rng.integers(0, 2, 200),
        }
    )

    df = extract_datetime_features(df, "signup_date", drop_original=True)
    df = bin_column(df, "age", bins=5)
    df = add_ratio_feature(df, "income", "age")
    df = log_transform(df, ["income"])
    print(df.head())
    print("Shape:", df.shape)
