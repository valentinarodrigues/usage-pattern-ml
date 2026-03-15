"""
Two-layer anomaly detection:

  Layer 1 — Service-level (Isolation Forest):
    Flags services whose overall feature profile is statistically unusual
    across the full observation window.

  Layer 2 — Time-window-level (Z-score):
    Flags individual hours where a service's metrics spike unexpectedly.
    Faster than Isolation Forest and interpretable for on-call engineers.
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SERVICE_FEATURES = ["requests_per_hour", "error_rate", "latency_p95", "peak_to_avg_ratio"]
TS_FEATURES = ["request_count", "error_rate", "latency_p50", "latency_p95"]


class AnomalyDetector:
    def __init__(self, config: dict):
        cfg = config["models"]["anomaly_detection"]
        self.method: str = cfg.get("method", "isolation_forest")
        self.contamination: float = cfg.get("contamination", 0.05)

        self.cache_dir = Path(config["processing"]["cache_dir"]) / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.model: Optional[IsolationForest] = None

    # ------------------------------------------------------------------
    # Service-level anomalies
    # ------------------------------------------------------------------

    def detect_service_anomalies(self, service_features: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in SERVICE_FEATURES if f in service_features.columns]
        X = service_features[available].fillna(0).values

        if len(X) < 10:
            logger.warning("[Anomaly] Too few services for detection — skipping")
            return service_features.assign(anomaly_score=0.0, is_anomaly=False)

        X_scaled = self.scaler.fit_transform(X)

        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                n_jobs=-1,
            )
            labels = self.model.fit_predict(X_scaled)
            # score_samples returns negative anomaly scores — flip so higher = more anomalous
            raw_scores = -self.model.score_samples(X_scaled)
        else:
            z = np.abs(stats.zscore(X_scaled, axis=0, nan_policy="omit"))
            raw_scores = z.max(axis=1)
            labels = np.where(raw_scores > 2.5, -1, 1)

        # Normalise scores to [0, 1]
        score_range = raw_scores.max() - raw_scores.min() + 1e-8
        normalised = (raw_scores - raw_scores.min()) / score_range

        result = service_features.copy()
        result["anomaly_score"] = normalised.round(4)
        result["is_anomaly"] = labels == -1

        n = int(result["is_anomaly"].sum())
        logger.info(f"[Anomaly] {n} anomalous services ({n/len(result)*100:.1f}%)")

        joblib.dump({"scaler": self.scaler, "model": self.model}, self.cache_dir / "anomaly.pkl")
        return result

    # ------------------------------------------------------------------
    # Time-window-level anomalies
    # ------------------------------------------------------------------

    def detect_timeseries_anomalies(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Per-service Z-score detection over the time dimension."""
        results = []

        for service, group in ts.groupby("service_name"):
            group = group.sort_values("time_bucket").copy()
            available = [f for f in TS_FEATURES if f in group.columns]

            if len(group) < 24 or len(available) == 0:
                continue

            X = group[available].fillna(0).astype(float).values
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy="omit"))
            group["anomaly_score"] = z_scores.max(axis=1).round(4)
            group["is_anomaly"] = group["anomaly_score"] > 3.0
            results.append(group)

        if not results:
            return pd.DataFrame()

        combined = pd.concat(results, ignore_index=True)
        n = int(combined["is_anomaly"].sum())
        logger.info(f"[Anomaly] {n} anomalous time windows detected")
        return combined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def top_anomalies(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        if "is_anomaly" not in df.columns:
            return pd.DataFrame()
        return df[df["is_anomaly"]].nlargest(n, "anomaly_score")
