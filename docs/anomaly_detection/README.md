# Anomaly Detection Documentation

## Overview

The `scripts/anomaly_detection/` package provides unsupervised anomaly detection for
network traffic and system log data.

---

## `network_anomaly_detector.py`

### Purpose
Detect unusual patterns in log entries without labelled training data.

### Log feature extraction

`extract_log_features(logs: pd.Series) → pd.DataFrame`

Converts raw log strings into a numeric feature matrix:

| Feature | Description |
|---|---|
| `log_length` | Character count |
| `word_count` | Whitespace-delimited token count |
| `severity_score` | Numeric log level (DEBUG=0 … FATAL=4) |
| `has_ip` | 1 if an IPv4 address is present |
| `has_error_keyword` | 1 if error-related words are found |
| `has_number` | 1 if any digit sequence appears |

### Detectors

#### `IsolationForestDetector`

Wraps `sklearn.ensemble.IsolationForest`.

```python
from scripts.anomaly_detection.network_anomaly_detector import IsolationForestDetector

det = IsolationForestDetector(contamination=0.05)
det.fit(features_df)
preds = det.predict(features_df)   # 1=normal, -1=anomaly
scores = det.anomaly_scores(features_df)
```

#### `LOFDetector`

Wraps `sklearn.neighbors.LocalOutlierFactor` in novelty-detection mode.

```python
from scripts.anomaly_detection.network_anomaly_detector import LOFDetector

det = LOFDetector(n_neighbors=20, contamination=0.05)
det.fit(features_df)
preds = det.predict(new_features_df)
```

### High-level `AnomalyDetector`

End-to-end pipeline: feature extraction + detection.

```python
from scripts.anomaly_detection.network_anomaly_detector import AnomalyDetector

detector = AnomalyDetector(method='isolation_forest', contamination=0.05)
results = detector.fit_detect(df_logs)     # adds is_anomaly & anomaly_score columns
anomalies = results[results['is_anomaly']]
```

#### Parameters

| Parameter | Description |
|---|---|
| `method` | `'isolation_forest'` or `'lof'` |
| `contamination` | Expected fraction of anomalies (0–0.5) |
| `log_column` | Name of the raw-text column (`None` if features are pre-computed) |

---

## Real-world applications

- **Intrusion detection**: Flag unusual SSH login patterns or port scans.
- **Application monitoring**: Detect service degradation from error-log spikes.
- **Infrastructure security**: Identify privilege escalations in system logs.
- **Network traffic analysis**: Spot DDoS patterns or exfiltration attempts.
