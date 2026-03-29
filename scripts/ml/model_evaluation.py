"""
Model Evaluation
================
Classification and regression metrics, confusion-matrix plotting, ROC/PR
curves, and feature-importance visualisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    """
    Return a dictionary of key classification metrics.

    Parameters
    ----------
    y_true : ground-truth labels
    y_pred : predicted labels
    y_prob : predicted probabilities for the positive class (binary tasks)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            pass
    logger.info("Classification metrics: %s", metrics)
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> None:
    """Print sklearn's full classification report."""
    print(classification_report(y_true, y_pred, target_names=labels))


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Return MAE, RMSE, and R² for regression tasks."""
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }
    logger.info("Regression metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> None:
    """Plot and optionally save the confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, colorbar=True, ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120)
        logger.info("Saved confusion matrix to %s", output_path)
    plt.close(fig)


def plot_roc_curve(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """Plot and optionally save the ROC curve for a binary classifier."""
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120)
        logger.info("Saved ROC curve to %s", output_path)
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """Plot and optionally save the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, marker=".", color="darkorange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120)
        logger.info("Saved PR curve to %s", output_path)
    plt.close(fig)


def plot_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str],
    top_n: int = 20,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot the top-*top_n* feature importances from a tree-based model
    embedded in a Pipeline (expects a ``model`` step).
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_.")
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color="steelblue",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_n} Feature Importances")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120)
        logger.info("Saved feature importances to %s", output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    from scripts.ml.model_training import build_pipeline

    X_arr, y_arr = make_classification(
        n_samples=600, n_features=10, n_informative=6, random_state=1
    )
    X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(10)])
    y_s = pd.Series(y_arr)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2)
    pipeline = build_pipeline("random_forest")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(classification_metrics(y_test, y_pred, y_prob))
    plot_confusion_matrix(y_test.values, y_pred)
