"""
Synthesises ML outputs into a structured report with four sections:

  1. traffic_summary      — top services, cluster breakdown, growth leaders
  2. anomaly_alerts       — services/windows with unusual behaviour
  3. capacity_plan        — forecast-driven scale recommendations
  4. cost_optimization    — actionable AWS + architectural cost savings

The cost_optimization section is the primary deliverable for engineering and POs.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, config: dict):
        self.top_n: int = config["reporting"].get("top_n_services", 20)
        self.thresholds: dict = config["reporting"].get("alert_thresholds", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        clustered_services: pd.DataFrame,
        forecasts: Dict[str, Optional[pd.DataFrame]],
        anomalies: pd.DataFrame,
        ts_anomalies: pd.DataFrame,
        growth_rates: pd.DataFrame,
        ts_features: pd.DataFrame,
        date_range: dict,
    ) -> Dict[str, Any]:

        report = {
            "metadata": self._metadata(clustered_services, date_range),
            "traffic_summary": self._traffic_summary(clustered_services, growth_rates),
            "anomaly_alerts": self._anomaly_alerts(anomalies, ts_anomalies),
            "capacity_plan": self._capacity_plan(forecasts, clustered_services),
            "cost_optimization": self._cost_optimization(
                clustered_services, ts_features, forecasts
            ),
        }

        self._log_summary(report)
        return report

    # ------------------------------------------------------------------
    # Report sections
    # ------------------------------------------------------------------

    def _metadata(self, services: pd.DataFrame, date_range: dict) -> dict:
        return {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "date_range": date_range,
            "total_services_analysed": int(len(services)),
            "total_requests": int(services["total_requests"].sum()),
        }

    def _traffic_summary(self, services: pd.DataFrame, growth_rates: pd.DataFrame) -> dict:
        top = services.nlargest(self.top_n, "total_requests")[
            [
                "service_name",
                "total_requests",
                "requests_per_hour",
                "error_rate",
                "latency_p95",
                "cluster_label",
            ]
        ]

        cluster_summary = (
            services.groupby("cluster_label")
            .agg(
                service_count=("service_name", "count"),
                total_requests=("total_requests", "sum"),
                avg_error_rate=("error_rate", "mean"),
                avg_latency_p95=("latency_p95", "mean"),
            )
            .reset_index()
            .sort_values("total_requests", ascending=False)
        )

        growth_threshold = self.thresholds.get("growth_rate_pct", 50)
        fast_growing = (
            growth_rates[growth_rates["growth_rate_pct"] > growth_threshold]
            .head(10)
            .to_dict(orient="records")
            if not growth_rates.empty
            else []
        )
        declining = (
            growth_rates[growth_rates["growth_rate_pct"] < -20]
            .head(10)
            .to_dict(orient="records")
            if not growth_rates.empty
            else []
        )

        return {
            "top_services": top.to_dict(orient="records"),
            "cluster_breakdown": cluster_summary.to_dict(orient="records"),
            "fast_growing_services": fast_growing,
            "declining_services": declining,
        }

    def _anomaly_alerts(
        self,
        service_anomalies: pd.DataFrame,
        ts_anomalies: pd.DataFrame,
    ) -> dict:
        score_threshold = self.thresholds.get("anomaly_score", 0.75)

        # Service-level alerts
        svc_alerts = []
        if not service_anomalies.empty and "is_anomaly" in service_anomalies.columns:
            flagged = service_anomalies[
                service_anomalies["is_anomaly"]
                & (service_anomalies["anomaly_score"] >= score_threshold)
            ].nlargest(10, "anomaly_score")

            for _, row in flagged.iterrows():
                svc_alerts.append(
                    {
                        "service": row["service_name"],
                        "anomaly_score": round(float(row["anomaly_score"]), 3),
                        "error_rate": round(float(row.get("error_rate", 0)), 3),
                        "latency_p95_ms": round(float(row.get("latency_p95", 0)), 1),
                        "recommendation": self._service_recommendation(row),
                    }
                )

        # Time-window alerts (top 10 most anomalous windows)
        window_alerts = []
        if not ts_anomalies.empty and "is_anomaly" in ts_anomalies.columns:
            flagged_windows = (
                ts_anomalies[ts_anomalies["is_anomaly"]]
                .nlargest(10, "anomaly_score")[
                    ["service_name", "time_bucket", "anomaly_score", "request_count", "error_rate"]
                ]
            )
            window_alerts = flagged_windows.to_dict(orient="records")

        return {"service_alerts": svc_alerts, "time_window_alerts": window_alerts}

    def _capacity_plan(
        self,
        forecasts: Dict[str, Optional[pd.DataFrame]],
        services: pd.DataFrame,
    ) -> dict:
        peak_windows = []
        scale_recommendations = []

        for service, fc in forecasts.items():
            if fc is None or fc.empty:
                continue

            peak_upper = float(fc["upper_bound"].max())
            peak_time = str(fc.loc[fc["upper_bound"].idxmax(), "time_bucket"])
            peak_windows.append(
                {
                    "service_name": service,
                    "peak_time": peak_time,
                    "peak_rph_upper": round(peak_upper, 1),
                }
            )

            svc_row = services[services["service_name"] == service]
            if svc_row.empty:
                continue
            current_rph = float(svc_row["requests_per_hour"].values[0])
            scale_factor = peak_upper / current_rph if current_rph > 0 else 1.0

            if scale_factor > 1.3:
                scale_recommendations.append(
                    {
                        "service": service,
                        "current_avg_rph": round(current_rph, 1),
                        "forecast_peak_rph": round(peak_upper, 1),
                        "recommended_scale_factor": round(scale_factor, 2),
                        "urgency": "high" if scale_factor > 2.5 else "medium",
                        "action": f"Pre-scale by {scale_factor:.1f}× before forecasted peak at {peak_time}",
                    }
                )

        scale_recommendations.sort(key=lambda x: x["recommended_scale_factor"], reverse=True)
        peak_windows.sort(key=lambda x: x["peak_rph_upper"], reverse=True)

        return {
            "forecast_peak_windows": peak_windows[:20],
            "scale_recommendations": scale_recommendations,
        }

    def _cost_optimization(
        self,
        services: pd.DataFrame,
        ts: pd.DataFrame,
        forecasts: Dict[str, Optional[pd.DataFrame]],
    ) -> dict:
        """
        Actionable cost-saving recommendations grouped by category.
        Each item includes: finding, action, estimated_saving, and optional implementation hint.
        """
        insights: dict = {
            "aws_compute": [],
            "aws_storage": [],
            "data_processing": [],
            "architectural": [],
        }

        # ---- AWS COMPUTE ------------------------------------------------

        # Idle services → Lambda / consolidation
        idle = services[services["requests_per_hour"] < 1.0]
        if not idle.empty:
            insights["aws_compute"].append(
                {
                    "category": "Idle Services",
                    "finding": f"{len(idle)} services average <1 req/hour",
                    "action": (
                        "Migrate to AWS Lambda (pay-per-request) or consolidate into a "
                        "shared service. Eliminate always-on compute for near-zero traffic."
                    ),
                    "estimated_saving": "HIGH — eliminates 100% of dedicated instance cost for idle services",
                    "sample_services": idle["service_name"].tolist()[:8],
                }
            )

        # Over-provisioned (flat traffic, no burst)
        if "peak_to_avg_ratio" in services.columns:
            over_prov = services[
                (services["peak_to_avg_ratio"] < 1.5) & (services["requests_per_hour"] < 100)
            ]
            if not over_prov.empty:
                insights["aws_compute"].append(
                    {
                        "category": "Over-provisioned (Flat Traffic)",
                        "finding": f"{len(over_prov)} services have peak/avg < 1.5× — no real burst",
                        "action": (
                            "Right-size to smaller instance types. "
                            "Flat traffic means large burst capacity is wasted."
                        ),
                        "estimated_saving": "MEDIUM — typically 25–40% compute cost reduction",
                    }
                )

        # Spot instance candidates (weekend-quiet = batch/internal)
        if "is_weekend" in ts.columns:
            total_by_svc = ts.groupby("service_name")["request_count"].sum()
            weekend_by_svc = ts[ts["is_weekend"] == 1].groupby("service_name")["request_count"].sum()
            weekend_ratio = (weekend_by_svc / total_by_svc.replace(0, np.nan)).fillna(0)
            spot_candidates = weekend_ratio[weekend_ratio < 0.1].index.tolist()
            if spot_candidates:
                insights["aws_compute"].append(
                    {
                        "category": "Spot Instance Candidates",
                        "finding": f"{len(spot_candidates)} services have <10% weekend traffic (batch / internal)",
                        "action": (
                            "Run batch, ETL, and non-user-facing services on EC2 Spot Instances "
                            "or AWS Fargate Spot."
                        ),
                        "estimated_saving": "HIGH — Spot is 60–90% cheaper than On-Demand",
                        "sample_services": spot_candidates[:8],
                    }
                )

        # Under-provisioned (proactive scale-out)
        under_prov_services = []
        for service, fc in forecasts.items():
            if fc is None or fc.empty:
                continue
            svc_row = services[services["service_name"] == service]
            if svc_row.empty:
                continue
            current = float(svc_row["requests_per_hour"].values[0])
            peak = float(fc["upper_bound"].max())
            if current > 0 and peak / current > 2.5:
                under_prov_services.append(service)

        if under_prov_services:
            insights["aws_compute"].append(
                {
                    "category": "Pre-emptive Auto-Scaling",
                    "finding": f"{len(under_prov_services)} services are forecast to spike >2.5× current load",
                    "action": (
                        "Configure predictive auto-scaling (AWS Auto Scaling predictive policies) "
                        "triggered 5–10 min before forecasted peaks. Avoids cold-start latency spikes."
                    ),
                    "estimated_saving": "MEDIUM — reduces over-provisioning buffer by 20–30%",
                    "sample_services": under_prov_services[:5],
                }
            )

        # ---- AWS STORAGE ------------------------------------------------

        insights["aws_storage"].append(
            {
                "category": "Log Format: CSV → Parquet",
                "finding": "Raw CSV/JSON logs are expensive to store and costly for Athena to query",
                "action": (
                    "Convert raw logs to Parquet (Snappy) using an AWS Glue job. "
                    "Schedule daily conversion via EventBridge."
                ),
                "estimated_saving": "HIGH — ~75% storage reduction; ~80% Athena query cost reduction",
                "implementation": (
                    "aws glue create-job --name raw-to-parquet "
                    "--role GlueServiceRole --command Name=glueetl,ScriptLocation=s3://..."
                ),
            }
        )

        insights["aws_storage"].append(
            {
                "category": "S3 Hive Partitioning",
                "finding": "Unpartitioned logs force Athena to scan the full dataset for every query",
                "action": (
                    "Partition S3 paths as: s3://bucket/logs/year=YYYY/month=MM/day=DD/ "
                    "Register partitions with AWS Glue Catalog or run MSCK REPAIR TABLE."
                ),
                "estimated_saving": "HIGH — reduces per-query scan by 70–95% for date-filtered queries",
                "implementation": "ALTER TABLE access_logs ADD PARTITION (dt='2025-01-15') LOCATION 's3://...'",
            }
        )

        insights["aws_storage"].append(
            {
                "category": "S3 Lifecycle Policy",
                "finding": "Logs older than 90 days remain in S3 Standard (expensive)",
                "action": (
                    "Apply a lifecycle rule: transition to S3 Intelligent-Tiering at 30 days, "
                    "Glacier Instant Retrieval at 90 days, expire at 365 days."
                ),
                "estimated_saving": "MEDIUM — Intelligent-Tiering saves 40–68% on infrequent access; Glacier ~80%",
                "implementation": (
                    "aws s3api put-bucket-lifecycle-configuration "
                    "--bucket your-logs-bucket --lifecycle-configuration file://lifecycle.json"
                ),
            }
        )

        insights["aws_storage"].append(
            {
                "category": "Athena Workgroup Cost Controls",
                "finding": "Runaway queries with no scan limit can incur unexpected costs",
                "action": (
                    "Set per-query data scan limit on the Athena workgroup. "
                    "Also enable query result reuse (TTL=24h) to avoid re-billing identical queries."
                ),
                "estimated_saving": "MEDIUM — prevents cost spikes; result reuse eliminates repeat scan charges",
                "implementation": (
                    "aws athena update-work-group --work-group ml-usage-analysis "
                    "--configuration ResultConfiguration={...},BytesScannedCutoffPerQuery=1073741824"
                ),
            }
        )

        # ---- DATA PROCESSING --------------------------------------------

        insights["data_processing"].append(
            {
                "category": "Incremental Pipeline",
                "finding": "Reprocessing full history on every run is the #1 pipeline cost driver",
                "action": (
                    "This pipeline already runs incrementally (state file tracks last processed date). "
                    "Ensure Glue jobs and Athena queries also use partition-pruned incremental patterns."
                ),
                "estimated_saving": "HIGH — reduces daily Athena scan volume by 95%+ vs full table scans",
            }
        )

        insights["data_processing"].append(
            {
                "category": "Downsampling for Model Training",
                "finding": "ML models don't need every single log record to learn accurate patterns",
                "action": (
                    "Set sample_rate: 0.1–0.3 in config for clustering and anomaly detection. "
                    "Use full data only for forecasting (time series needs continuity)."
                ),
                "estimated_saving": "MEDIUM — 70–90% reduction in feature engineering compute time",
            }
        )

        insights["data_processing"].append(
            {
                "category": "Model Caching",
                "finding": "Clustering and anomaly models are retrained on every run by default",
                "action": (
                    "Cache trained models (joblib). Only retrain when data volume changes >10% "
                    "or on a weekly schedule. Warm-load models for daily runs."
                ),
                "estimated_saving": "MEDIUM — eliminates redundant training compute on unchanged patterns",
            }
        )

        insights["data_processing"].append(
            {
                "category": "Off-Peak Scheduling",
                "finding": "Pipeline runs during business hours compete with interactive workloads",
                "action": (
                    "Schedule the pipeline 02:00–05:00 UTC via EventBridge + Step Functions. "
                    "Spot instance availability is highest in this window."
                ),
                "estimated_saving": "LOW-MEDIUM — 15–30% lower Spot prices; no resource contention",
            }
        )

        # ---- ARCHITECTURAL ----------------------------------------------

        # High-traffic services → API standardisation / reusability
        high_traffic = services[services["requests_per_hour"] > 100]
        if not high_traffic.empty:
            insights["architectural"].append(
                {
                    "category": "Reusability Candidates",
                    "finding": f"{len(high_traffic)} services each receive >100 req/hour",
                    "action": (
                        "Prioritise these services for shared API layer / data product standardisation. "
                        "High-traffic shared services reduce duplicate integrations across teams."
                    ),
                    "estimated_saving": "STRATEGIC — reduces duplicate build cost and integration effort",
                    "sample_services": high_traffic.nlargest(8, "requests_per_hour")[
                        "service_name"
                    ].tolist(),
                }
            )

        # Declining services → deprecation
        if not services.empty and "total_requests" in services.columns:
            very_low = services[
                (services["total_requests"] < 100) & (services["error_rate"] > 0.2)
            ]
            if not very_low.empty:
                insights["architectural"].append(
                    {
                        "category": "Deprecation Candidates",
                        "finding": f"{len(very_low)} services have <100 total requests AND >20% error rate",
                        "action": (
                            "Review for deprecation or replacement. "
                            "High error rates on low-traffic services often signal abandoned services."
                        ),
                        "estimated_saving": "MEDIUM — removes maintenance burden and idle compute",
                        "sample_services": very_low["service_name"].tolist()[:8],
                    }
                )

        insights["architectural"].append(
            {
                "category": "Caching Layer (CDN / ElastiCache)",
                "finding": "High-volume read endpoints with low latency variance are cacheable",
                "action": (
                    "Identify endpoints with GET methods, status 200, and low latency std-dev. "
                    "Add CloudFront or ElastiCache in front to absorb repeat requests."
                ),
                "estimated_saving": "HIGH — cache hit rate of 60%+ reduces origin load and compute cost",
            }
        )

        return insights

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _service_recommendation(self, row: pd.Series) -> str:
        if float(row.get("error_rate", 0)) > 0.15:
            return "High error rate — investigate health, dependencies, and recent deployments"
        if float(row.get("latency_p95", 0)) > 2000:
            return "High P95 latency — check DB bottlenecks, connection pools, or downstream timeouts"
        if float(row.get("requests_per_hour", 0)) > 10000:
            return "Unexpected traffic volume — verify for DDoS, runaway client, or missing rate limit"
        return "Anomalous profile — review recent config changes and compare with baseline"

    def _log_summary(self, report: dict):
        m = report["metadata"]
        logger.info(f"  Services: {m['total_services_analysed']}  |  Requests: {m['total_requests']:,}")
        logger.info(f"  Anomaly alerts: {len(report['anomaly_alerts']['service_alerts'])}")
        logger.info(
            f"  Scale recommendations: {len(report['capacity_plan']['scale_recommendations'])}"
        )
        total_insights = sum(len(v) for v in report["cost_optimization"].values())
        logger.info(f"  Cost insights: {total_insights}")
