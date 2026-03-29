"""
Preprocessing
=============
Transformations applied after data cleaning: encoding, scaling, and
train/test splitting – ready for ML model ingestion.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def label_encode(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply label encoding to categorical columns.

    Returns
    -------
    df_encoded : pd.DataFrame
    encoders : dict mapping column name to its fitted LabelEncoder
    """
    df = df.copy()
    cat_cols = columns or df.select_dtypes(include=["str", "object", "category"]).columns.tolist()
    encoders: dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.debug("Label-encoded '%s' (%d classes).", col, len(le.classes_))
    return df, encoders


def one_hot_encode(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame with one-hot encoded columns."""
    cat_cols = columns or df.select_dtypes(include=["str", "object", "category"]).columns.tolist()
    return pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def standard_scale(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise numeric columns (zero mean, unit variance).

    Returns the transformed DataFrame and the fitted scaler.
    """
    df = df.copy()
    num_cols = columns or df.select_dtypes(include="number").columns.tolist()
    scaler = scaler or StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


def minmax_scale(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    feature_range: Tuple[float, float] = (0, 1),
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric columns to *feature_range* using Min-Max normalisation.

    Returns the transformed DataFrame and the fitted scaler.
    """
    df = df.copy()
    num_cols = columns or df.select_dtypes(include="number").columns.tolist()
    scaler = scaler or MinMaxScaler(feature_range=feature_range)
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix *X* from the target column *y*."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_test_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple:
    """
    Split data into train, validation, and test sets.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=1 - val_ratio, random_state=random_state
    )
    logger.info(
        "Split sizes – train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, 300),
            "income": rng.exponential(40_000, 300),
            "city": rng.choice(["NY", "LA", "SF"], 300),
            "churn": rng.integers(0, 2, 300),
        }
    )

    df_ohe = one_hot_encode(df, columns=["city"])
    X, y = split_features_target(df_ohe, target_column="churn")
    X_scaled, scaler = standard_scale(X)
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(X_scaled, y)
    print("X_train shape:", X_train.shape)
    print("Scaler mean:", scaler.mean_)
