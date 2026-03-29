"""
Model Training
==============
Generic training utilities that support multiple classifier / regressor
families with cross-validation, hyper-parameter tuning, and persistence.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Registry of supported models and their default hyper-parameter grids
MODEL_REGISTRY: Dict[str, Tuple[Any, Dict]] = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [3, 5],
        },
    ),
}


def build_pipeline(model_name: str) -> Pipeline:
    """
    Build a sklearn Pipeline consisting of StandardScaler → model.

    Parameters
    ----------
    model_name : key in MODEL_REGISTRY

    Returns
    -------
    sklearn Pipeline
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    estimator, _ = MODEL_REGISTRY[model_name]
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def train_with_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
    cv: int = 5,
    scoring: str = "f1_weighted",
) -> Tuple[Pipeline, np.ndarray]:
    """
    Train *model_name* with stratified k-fold CV and return the pipeline
    fitted on the full training data along with per-fold scores.

    Returns
    -------
    pipeline : fitted Pipeline
    cv_scores : array of per-fold metric values
    """
    pipeline = build_pipeline(model_name)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring)
    logger.info(
        "%s CV %s: %.4f ± %.4f",
        model_name,
        scoring,
        cv_scores.mean(),
        cv_scores.std(),
    )
    pipeline.fit(X, y)
    return pipeline, cv_scores


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
    cv: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Perform grid-search hyper-parameter tuning.

    Returns the fitted GridSearchCV object (best estimator accessible via
    ``result.best_estimator_``).
    """
    pipeline = build_pipeline(model_name)
    _, param_grid = MODEL_REGISTRY[model_name]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X, y)
    logger.info(
        "Best params: %s | Best %s: %.4f",
        grid_search.best_params_,
        scoring,
        grid_search.best_score_,
    )
    return grid_search


def save_model(pipeline: Pipeline, path: str) -> None:
    """Persist a fitted pipeline to disk using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Model saved to %s", path)


def load_model(path: str) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model file found at '{path}'")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from sklearn.datasets import make_classification

    X_arr, y_arr = make_classification(
        n_samples=800, n_features=15, n_informative=8, random_state=0
    )
    X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
    y_s = pd.Series(y_arr, name="target")

    for name in MODEL_REGISTRY:
        pipeline, scores = train_with_cross_validation(X_df, y_s, model_name=name)
        print(f"{name}: mean CV F1 = {scores.mean():.4f}")

    save_model(pipeline, "/tmp/models/demo_model.joblib")
    loaded = load_model("/tmp/models/demo_model.joblib")
    print("Loaded model type:", type(loaded))
