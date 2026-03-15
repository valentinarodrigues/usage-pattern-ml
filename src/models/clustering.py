"""
Segments services into usage-profile clusters.

Cluster insight drives:
  - Auto-scaling policy selection (spiky clusters need HPA / target tracking)
  - Idle service candidates for deprecation or Lambda migration
  - Reusability prioritisation (high-traffic clusters deserve shared API layers)
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Features used for clustering — order matters (kept consistent across fit/predict)
CLUSTER_FEATURES = [
    "requests_per_hour",
    "error_rate",
    "latency_p95",
    "peak_to_avg_ratio",
    "unique_users",
    "unique_endpoints",
]

# Human-readable labels assigned by post-hoc centroid analysis
# Overridden by describe_clusters() output which is data-driven
CLUSTER_LABEL_MAP = {
    0: "high_traffic_stable",
    1: "high_traffic_spiky",
    2: "medium_traffic",
    3: "low_traffic_reliable",
    4: "low_traffic_error_prone",
    5: "bursty",
    6: "idle",
    7: "internal_api",
}


class ServiceClusterer:
    def __init__(self, config: dict):
        cfg = config["models"]["clustering"]
        self.algorithm: str = cfg.get("algorithm", "kmeans")
        self.n_clusters: int = cfg.get("n_clusters", 8)
        self.auto_k: bool = cfg.get("auto_k", False)
        self.dbscan_eps: float = cfg.get("dbscan_eps", 0.5)
        self.dbscan_min_samples: int = cfg.get("dbscan_min_samples", 5)

        self.cache_dir = Path(config["processing"]["cache_dir"]) / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.model: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, service_features: pd.DataFrame) -> pd.DataFrame:
        """Fit model and return service_features with cluster_id / cluster_label columns."""
        X, _ = self._extract_features(service_features)
        X_scaled = self.scaler.fit_transform(X)

        if self.algorithm == "dbscan":
            self.model = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            labels = self.model.fit_predict(X_scaled)
        else:
            k = self._optimal_k(X_scaled) if self.auto_k else self.n_clusters
            self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = self.model.fit_predict(X_scaled)

        # Silhouette score (skip if DBSCAN produced noise points or only 1 cluster)
        unique_labels = set(labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            score = silhouette_score(X_scaled, labels, sample_size=min(1000, len(X_scaled)))
            logger.info(f"[Clustering] Silhouette score: {score:.3f}")

        result = service_features.copy()
        result["cluster_id"] = labels
        result["cluster_label"] = result["cluster_id"].map(CLUSTER_LABEL_MAP).fillna("other")

        self._save()
        logger.info(f"[Clustering] {len(service_features)} services → {len(unique_labels)} clusters")
        return result

    def describe_clusters(self, clustered: pd.DataFrame) -> pd.DataFrame:
        """Data-driven cluster summaries for the report."""
        summary = (
            clustered.groupby(["cluster_id", "cluster_label"])
            .agg(
                service_count=("service_name", "count"),
                avg_rph=("requests_per_hour", "mean"),
                avg_error_rate=("error_rate", "mean"),
                avg_latency_p95=("latency_p95", "mean"),
                sample_services=("service_name", lambda x: list(x)[:5]),
            )
            .reset_index()
            .sort_values("avg_rph", ascending=False)
        )
        return summary

    def load(self) -> bool:
        path = self.cache_dir / "clusterer.pkl"
        if path.exists():
            saved = joblib.load(path)
            self.scaler, self.model = saved["scaler"], saved["model"]
            logger.info("[Clustering] Loaded cached model")
            return True
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame):
        available = [f for f in CLUSTER_FEATURES if f in df.columns]
        return df[available].fillna(0).values, available

    def _optimal_k(self, X: np.ndarray, k_range=range(3, 13)) -> int:
        best_k, best_score = 4, -1.0
        for k in k_range:
            if k >= len(X):
                break
            labels = KMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(X)
            score = silhouette_score(X, labels, sample_size=min(500, len(X)))
            if score > best_score:
                best_score, best_k = score, k
        logger.info(f"[Clustering] Auto K={best_k} (silhouette={best_score:.3f})")
        return best_k

    def _save(self):
        joblib.dump(
            {"scaler": self.scaler, "model": self.model},
            self.cache_dir / "clusterer.pkl",
        )
