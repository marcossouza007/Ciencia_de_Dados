"""
Anomaly Detection for Network and System Logs
=============================================
Implements Isolation Forest and Local Outlier Factor detectors, a log-feature
extractor, and a unified AnomalyDetector class that wraps the full pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log feature extraction
# ---------------------------------------------------------------------------

# Common log patterns (syslog / access-log style)
LOG_PATTERN = re.compile(
    r"(?P<timestamp>\w{3}\s+\d+\s+\d+:\d+:\d+)?\s*"
    r"(?P<host>\S+)?\s*"
    r"(?P<process>[^\[]+)?\[?(?P<pid>\d+)?\]?:?\s*"
    r"(?P<level>ERROR|WARN|INFO|DEBUG|CRITICAL|FATAL)?\s*"
    r"(?P<message>.+)?",
    re.IGNORECASE,
)

SEVERITY_MAP = {
    "debug": 0,
    "info": 1,
    "warn": 2,
    "warning": 2,
    "error": 3,
    "critical": 4,
    "fatal": 4,
}


def extract_log_features(logs: pd.Series) -> pd.DataFrame:
    """
    Parse a Series of raw log strings into a feature DataFrame.

    Features extracted:
    - ``log_length``       : character count of each log entry
    - ``word_count``       : number of whitespace-delimited tokens
    - ``severity_score``   : numeric encoding of log level (0=DEBUG … 4=FATAL)
    - ``has_ip``           : 1 if the entry contains an IPv4 address
    - ``has_error_keyword``: 1 if the entry contains error-related words
    - ``has_number``       : 1 if the entry contains any digit sequence
    """
    ip_re = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
    error_kw_re = re.compile(
        r"\b(error|fail|exception|traceback|timeout|refused|denied|critical|fatal)\b",
        re.IGNORECASE,
    )
    level_re = re.compile(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\b", re.IGNORECASE)

    records = []
    for entry in logs:
        entry_str = str(entry)
        level_match = level_re.search(entry_str)
        level_str = level_match.group(1).lower() if level_match else "info"
        records.append(
            {
                "log_length": len(entry_str),
                "word_count": len(entry_str.split()),
                "severity_score": SEVERITY_MAP.get(level_str, 1),
                "has_ip": int(bool(ip_re.search(entry_str))),
                "has_error_keyword": int(bool(error_kw_re.search(entry_str))),
                "has_number": int(bool(re.search(r"\d+", entry_str))),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """
    Wraps sklearn's IsolationForest for anomaly detection.

    Attributes
    ----------
    contamination : expected proportion of anomalies (0–0.5)
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42, **kwargs):
        self.contamination = contamination
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            **kwargs,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._fitted = True
        logger.info("IsolationForestDetector fitted on %d samples.", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1 for normal, -1 for anomaly (sklearn convention)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw anomaly scores (lower = more anomalous)."""
        X_scaled = self._scaler.transform(X)
        return self._model.decision_function(X_scaled)


class LOFDetector:
    """
    Wraps sklearn's LocalOutlierFactor for novelty detection.

    Note: LOF in *novelty=True* mode is used so that predict() works on new
    data after fitting.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self._scaler = StandardScaler()
        self._model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            **kwargs,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "LOFDetector":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._fitted = True
        logger.info("LOFDetector fitted on %d samples.", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1 for normal, -1 for anomaly."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.decision_function(X_scaled)


# ---------------------------------------------------------------------------
# High-level AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    End-to-end anomaly detector for network / system log DataFrames.

    1. Extracts numeric features from raw log text (if a ``log_column`` is given).
    2. Trains the selected detector on the feature set.
    3. Annotates the DataFrame with ``is_anomaly`` and ``anomaly_score`` columns.

    Parameters
    ----------
    method : {'isolation_forest', 'lof'}
    contamination : expected fraction of anomalies
    log_column : name of the raw log-text column; set to None if features
                 are pre-computed.
    """

    METHODS = {"isolation_forest": IsolationForestDetector, "lof": LOFDetector}

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        log_column: Optional[str] = "log",
        **detector_kwargs,
    ):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {list(self.METHODS)}")
        self.method = method
        self.contamination = contamination
        self.log_column = log_column
        self._detector = self.METHODS[method](
            contamination=contamination, **detector_kwargs
        )

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.log_column and self.log_column in df.columns:
            return extract_log_features(df[self.log_column])
        return df.select_dtypes(include="number")

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        features = self._build_features(df)
        self._detector.fit(features)
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the input DataFrame annotated with:
        - ``is_anomaly`` (bool)
        - ``anomaly_score`` (float, lower = more anomalous)
        """
        df = df.copy()
        features = self._build_features(df)
        preds = self._detector.predict(features)
        scores = self._detector.anomaly_scores(features)
        df["is_anomaly"] = preds == -1
        df["anomaly_score"] = scores
        n_anomalies = df["is_anomaly"].sum()
        logger.info(
            "Detected %d anomalies out of %d records (%.1f %%).",
            n_anomalies,
            len(df),
            100 * n_anomalies / max(len(df), 1),
        )
        return df

    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit then detect on the same DataFrame."""
        return self.fit(df).detect(df)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate a mix of normal and anomalous log entries
    normal_logs = [
        "Jan  1 00:01:15 server01 sshd[1234]: INFO Accepted password for user from 192.168.1.1",
        "Jan  1 00:02:30 server01 cron[5678]: INFO (root) CMD (/usr/bin/backup.sh)",
        "Jan  1 00:03:45 server01 kernel: INFO eth0: Link is up at 1 Gbps",
    ] * 50

    anomaly_logs = [
        "CRITICAL server01 sshd[9999]: ERROR Failed password for invalid user admin from 10.0.0.1 – BRUTE FORCE DETECTED traceback exception timeout",
        "FATAL kernel: ERROR segfault at 0x0000deadbeef ip 0x00007f9a11223344 error 4 in libc.so",
    ] * 5

    all_logs = pd.DataFrame({"log": normal_logs + anomaly_logs})
    detector = AnomalyDetector(method="isolation_forest", contamination=0.1)
    results = detector.fit_detect(all_logs)

    print("Total anomalies detected:", results["is_anomaly"].sum())
    print(results[results["is_anomaly"]]["log"].head())
