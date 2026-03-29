# ML Pipeline Documentation

## Overview

The `scripts/ml/` package covers feature engineering, model training, hyper-parameter
tuning, and evaluation.

---

## `feature_engineering.py`

### Purpose
Enrich the feature set before training to improve model performance.

### Key functions

| Function | Description |
|---|---|
| `add_polynomial_features(df, columns, degree=2)` | Polynomial and interaction features using sklearn's `PolynomialFeatures`. |
| `extract_datetime_features(df, column)` | Decompose a datetime column into year, month, day, day-of-week, hour, is_weekend, and cyclical hour encoding. |
| `bin_column(df, column, bins=5)` | Bin numeric values into discrete intervals (`pd.cut`). |
| `add_ratio_feature(df, numerator, denominator)` | Safely compute a ratio feature with zero-division guard. |
| `log_transform(df, columns)` | Apply `log1p` to right-skewed columns. |
| `target_encode(df, categorical_columns, target_column)` | Smoothed target encoding to handle high-cardinality categoricals. |

---

## `model_training.py`

### Purpose
Train and persist classification models.

### Supported models (`MODEL_REGISTRY`)

| Key | Model |
|---|---|
| `logistic_regression` | `sklearn.linear_model.LogisticRegression` |
| `random_forest` | `sklearn.ensemble.RandomForestClassifier` |
| `gradient_boosting` | `sklearn.ensemble.GradientBoostingClassifier` |

All models are wrapped in a `Pipeline(StandardScaler → model)`.

### Key functions

| Function | Description |
|---|---|
| `build_pipeline(model_name)` | Create an unfitted `Pipeline`. |
| `train_with_cross_validation(X, y, model_name, cv=5)` | Train with stratified k-fold CV; returns `(pipeline, cv_scores)`. |
| `tune_hyperparameters(X, y, model_name, cv=5)` | `GridSearchCV` over the model's default param grid. |
| `save_model(pipeline, path)` | Persist using `joblib`. |
| `load_model(path)` | Load a previously saved pipeline. |

### Usage

```python
from scripts.ml.model_training import train_with_cross_validation, save_model

pipeline, scores = train_with_cross_validation(X_train, y_train, model_name='random_forest')
print(f'CV F1: {scores.mean():.4f}')
save_model(pipeline, 'models/churn_rf.joblib')
```

---

## `model_evaluation.py`

### Purpose
Compute and visualise classification / regression metrics.

### Key functions

| Function | Description |
|---|---|
| `classification_metrics(y_true, y_pred, y_prob)` | Returns dict with accuracy, F1-weighted, F1-macro, ROC-AUC. |
| `print_classification_report(y_true, y_pred)` | Full sklearn classification report. |
| `regression_metrics(y_true, y_pred)` | MAE, RMSE, R². |
| `plot_confusion_matrix(y_true, y_pred)` | Matplotlib confusion matrix heatmap. |
| `plot_roc_curve(estimator, X_test, y_test)` | ROC curve for binary classifiers. |
| `plot_precision_recall_curve(y_true, y_prob)` | PR curve. |
| `plot_feature_importances(pipeline, feature_names)` | Top-N feature importances for tree models. |

---

## Real-world applications

- **Customer churn prediction** (telecom, SaaS)
- **Credit risk scoring** (banking)
- **Predictive maintenance** (manufacturing)
- **Demand forecasting** (retail)
