"""
Transforms normalized log records into two feature matrices:

  service_features  — one row per service, aggregated over the full window.
                       Used by clustering and anomaly detection.

  timeseries_features — one row per (service, hour), with a complete time grid.
                         Used by forecasting and trend analysis.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _pct(series: pd.Series, q: float) -> float:
    vals = series.dropna()
    return float(np.percentile(vals, q)) if len(vals) else 0.0


class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------
    # Service-level features (for clustering / anomaly detection)
    # ------------------------------------------------------------------

    def build_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("[Features] Building service feature matrix")

        agg = (
            df.groupby("service_name")
            .agg(
                total_requests=("status_code", "count"),
                error_rate=("status_code", lambda x: (x >= 400).mean()),
                server_error_rate=("status_code", lambda x: (x >= 500).mean()),
                latency_p50=("latency_ms", lambda x: _pct(x, 50)),
                latency_p95=("latency_ms", lambda x: _pct(x, 95)),
                latency_p99=("latency_ms", lambda x: _pct(x, 99)),
                avg_bytes=("bytes_response", "mean"),
                unique_users=("user_id", "nunique"),
                unique_endpoints=("endpoint", "nunique"),
                unique_instances=("instance_id", "nunique"),
            )
            .reset_index()
        )

        # Active hours and requests-per-hour
        span = (
            df.groupby("service_name")["timestamp"]
            .agg(first="min", last="max")
            .reset_index()
        )
        span["active_hours"] = (
            (span["last"] - span["first"]).dt.total_seconds() / 3600
        ).clip(lower=1)
        agg = agg.merge(span[["service_name", "active_hours"]], on="service_name", how="left")
        agg["requests_per_hour"] = agg["total_requests"] / agg["active_hours"]

        # Peak-to-average ratio — high values flag spiky / bursty services
        hourly_counts = (
            df.groupby(["service_name", df["timestamp"].dt.floor("h")])
            .size()
            .reset_index(name="cnt")
        )
        peak_ratio = (
            hourly_counts.groupby("service_name")["cnt"]
            .agg(lambda x: x.max() / x.mean() if x.mean() > 0 else 1.0)
            .rename("peak_to_avg_ratio")
            .reset_index()
        )
        agg = agg.merge(peak_ratio, on="service_name", how="left")

        agg["avg_bytes"] = agg["avg_bytes"].fillna(0)
        logger.info(f"[Features] Service matrix: {len(agg)} rows × {agg.shape[1]} cols")
        return agg

    # ------------------------------------------------------------------
    # Time-series features (for forecasting)
    # ------------------------------------------------------------------

    def build_timeseries_features(self, df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
        logger.info(f"[Features] Building time-series features (freq='{freq}')")

        df = df.copy()
        df["time_bucket"] = df["timestamp"].dt.floor(freq)

        ts = (
            df.groupby(["service_name", "time_bucket"])
            .agg(
                request_count=("status_code", "count"),
                error_rate=("status_code", lambda x: (x >= 400).mean()),
                latency_p50=("latency_ms", lambda x: _pct(x, 50)),
                latency_p95=("latency_ms", lambda x: _pct(x, 95)),
                bytes_total=("bytes_response", "sum"),
                unique_users=("user_id", "nunique"),
            )
            .reset_index()
        )

        # Fill a complete, gap-free time grid per service (zeros = no traffic)
        all_buckets = pd.date_range(
            ts["time_bucket"].min(), ts["time_bucket"].max(), freq=freq, tz="UTC"
        )
        services = ts["service_name"].unique()
        full_index = pd.MultiIndex.from_product(
            [services, all_buckets], names=["service_name", "time_bucket"]
        )
        ts = (
            ts.set_index(["service_name", "time_bucket"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        # Temporal features (after reindex)
        ts["hour_of_day"] = ts["time_bucket"].dt.hour
        ts["day_of_week"] = ts["time_bucket"].dt.dayofweek
        ts["is_weekend"] = ts["day_of_week"].isin([5, 6]).astype(int)
        ts["is_business_hours"] = ts["hour_of_day"].between(9, 17).astype(int)
        ts["week_of_year"] = ts["time_bucket"].dt.isocalendar().week.astype(int)

        logger.info(
            f"[Features] Time series: {len(services)} services × {len(all_buckets)} buckets"
        )
        return ts

    # ------------------------------------------------------------------
    # Growth rates
    # ------------------------------------------------------------------

    def compute_growth_rates(self, ts: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
        """Week-over-week request growth per service."""
        window = window_days * 24  # hours
        rows = []
        for service, group in ts.groupby("service_name"):
            group = group.sort_values("time_bucket")
            if len(group) < window * 2:
                continue
            recent = group.tail(window)["request_count"].sum()
            prior = group.iloc[-window * 2 : -window]["request_count"].sum()
            if prior > 0:
                rows.append(
                    {
                        "service_name": service,
                        "recent_requests": int(recent),
                        "prior_requests": int(prior),
                        "growth_rate_pct": round((recent - prior) / prior * 100, 2),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["service_name", "recent_requests", "prior_requests", "growth_rate_pct"])
        return pd.DataFrame(rows).sort_values("growth_rate_pct", ascending=False)
