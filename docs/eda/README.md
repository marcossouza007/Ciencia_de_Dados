# EDA Module Documentation

## Overview

The `scripts/eda/` package provides three modules for cleaning raw data, transforming it into a
model-ready format, and exploring it visually.

---

## `data_cleaning.py`

### Purpose
Detect and handle data quality issues before modelling.

### Key functions

| Function | Description |
|---|---|
| `report_missing(df)` | Returns a DataFrame with missing-value counts and percentages per column. |
| `drop_high_missing_columns(df, threshold=0.5)` | Drops columns with a missing-value ratio above `threshold`. |
| `impute_numeric(df, strategy='median')` | Fills missing numeric values using `median`, `mean`, or `zero`. |
| `impute_categorical(df, fill_value='Unknown')` | Fills missing categorical/object values with `fill_value`. |
| `remove_duplicates(df, subset=None)` | Removes exact duplicate rows. |
| `clip_outliers_iqr(df, factor=1.5)` | Clips numeric outliers to [Q1−1.5×IQR, Q3+1.5×IQR]. |
| `flag_outliers_zscore(df, threshold=3.0)` | Adds boolean `<col>_is_outlier` columns based on Z-score. |
| `cast_dtypes(df, dtype_map)` | Casts columns to the specified dtypes. |
| `clean_pipeline(df)` | Runs the complete opinionated cleaning pipeline. |

### Usage

```python
from scripts.eda.data_cleaning import report_missing, clean_pipeline

df_clean = clean_pipeline(df_raw)
print(report_missing(df_clean))   # Should be empty after cleaning
```

---

## `preprocessing.py`

### Purpose
Encode categorical variables, scale numeric features, and split the dataset into
train / validation / test sets.

### Key functions

| Function | Description |
|---|---|
| `label_encode(df, columns)` | Label-encodes categorical columns; returns `(df, encoders_dict)`. |
| `one_hot_encode(df, columns, drop_first=True)` | One-hot encodes categorical columns via `pd.get_dummies`. |
| `standard_scale(df, columns)` | Standardises numeric columns; returns `(df, StandardScaler)`. |
| `minmax_scale(df, columns, feature_range=(0,1))` | Min-max normalises numeric columns; returns `(df, MinMaxScaler)`. |
| `split_features_target(df, target_column)` | Splits the DataFrame into `X` and `y`. |
| `train_test_val_split(X, y, test_size=0.2, val_size=0.1)` | Three-way stratified split. |

### Usage

```python
from scripts.eda.preprocessing import one_hot_encode, standard_scale, train_test_val_split

df_ohe = one_hot_encode(df_clean, columns=['city', 'category'])
X, y   = split_features_target(df_ohe, target_column='churn')
X_sc, scaler = standard_scale(X)
X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(X_sc, y)
```

---

## `exploratory_analysis.py`

### Purpose
Provide visual and statistical insights into a dataset.

### Key functions

| Function | Description |
|---|---|
| `describe_dataframe(df)` | Extended `describe()` including skewness and kurtosis. |
| `value_counts_all(df, top_n=10)` | Value counts for all categorical columns. |
| `plot_distributions(df, output_dir)` | Histogram + KDE for each numeric column. |
| `plot_correlation_heatmap(df, output_path)` | Seaborn correlation heatmap. |
| `plot_boxplots(df, output_dir)` | Box plots for outlier visualisation. |
| `plot_target_distribution(df, target_column)` | Bar chart of class balance. |
| `generate_eda_report(df, output_dir, target_column)` | Full EDA report in one call. |

### Usage

```python
from scripts.eda.exploratory_analysis import generate_eda_report

generate_eda_report(df_clean, output_dir='reports/eda', target_column='churn')
```

All plots are saved as PNG files under `output_dir`.

---

## Real-world applications

- **Customer churn prediction**: Identify missing fields and class imbalance early.
- **Fraud detection**: Flag statistical outliers in transaction amounts.
- **Healthcare analytics**: Handle high-missing columns (e.g. optional lab results).
