"""
Local demo — runs the full pipeline on synthetic log data without AWS.
No AWS credentials required.

Usage:
  python demo_local.py
"""
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-35s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("demo")

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

SERVICES = [
    ("payment-service",       500, 0.02, 80),
    ("user-auth",             1200, 0.01, 30),
    ("product-catalog",       3000, 0.005, 20),
    ("order-service",         400,  0.03, 150),
    ("inventory-api",         200,  0.04, 200),
    ("notification-service",  50,   0.10, 50),
    ("legacy-reports",        2,    0.30, 800),    # idle + error-prone
    ("search-service",        800,  0.01, 40),
    ("recommendation-engine", 600,  0.02, 300),
    ("analytics-ingest",      5,    0.05, 120),    # near-idle
    ("billing-api",           150,  0.02, 90),
    ("admin-portal",          10,   0.08, 60),
    ("content-delivery",      5000, 0.001, 15),    # very high traffic
    ("data-export-job",       3,    0.01, 400),    # batch
    ("healthcheck-svc",       2000, 0.00, 2),      # high-freq, instant
]

ENDPOINTS = ["/api/v1/list", "/api/v1/get/{id}", "/api/v1/create", "/api/v1/update/{id}", "/health"]
METHODS = ["GET", "GET", "GET", "POST", "PUT", "DELETE", "GET"]
USERS = [f"user_{i}" for i in range(500)]


def make_synthetic_logs(hours: int = 168) -> pd.DataFrame:  # 7 days
    rng = np.random.default_rng(42)
    rows = []
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=hours)

    for svc, base_rph, err_rate, base_lat in SERVICES:
        for h in range(hours):
            ts_base = start + timedelta(hours=h)
            hour_of_day = ts_base.hour
            day_of_week = ts_base.weekday()

            # Simulate daily + weekly seasonality
            hour_factor = 1.0 + 0.8 * np.sin((hour_of_day - 9) * np.pi / 12)
            hour_factor = max(0.05, hour_factor)
            weekend_factor = 0.3 if day_of_week >= 5 else 1.0
            n_requests = max(0, int(rng.poisson(base_rph * hour_factor * weekend_factor)))

            for _ in range(n_requests):
                is_error = rng.random() < err_rate
                status = rng.choice([400, 404, 500, 503]) if is_error else rng.choice([200, 200, 200, 201, 204])
                lat = max(1, rng.normal(base_lat, base_lat * 0.3))
                if is_error:
                    lat *= rng.uniform(1.5, 4.0)

                rows.append(
                    {
                        "timestamp": ts_base + timedelta(seconds=int(rng.integers(0, 3600))),
                        "service_name": svc,
                        "endpoint": rng.choice(ENDPOINTS),
                        "method": rng.choice(METHODS),
                        "status_code": status,
                        "latency_ms": round(lat, 1),
                        "bytes_response": int(rng.integers(200, 50000)),
                        "user_id": rng.choice(USERS),
                        "instance_id": f"{svc}-pod-{rng.integers(1, 4)}",
                    }
                )

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df):,} synthetic log records across {len(SERVICES)} services")
    return df


# ---------------------------------------------------------------------------
# Minimal config (no AWS)
# ---------------------------------------------------------------------------

LOCAL_CONFIG = {
    "aws": {
        "region": "us-east-1",
        "s3": {
            "logs_bucket": "local-demo",
            "logs_prefix": "logs/",
            "parquet_prefix": "parquet/",
            "results_prefix": "results/",
        },
        "athena": {
            "database": "logs_db",
            "workgroup": "demo",
            "query_results_bucket": "s3://local/",
            "max_scan_mb": 9999,
            "table": "access_logs",
        },
        "glue": {"catalog_database": "logs_db"},
    },
    "processing": {
        "lookback_days": 7,
        "incremental": False,
        "cache_dir": ".cache_demo/",
        "sample_rate": 1.0,
        "parquet_compression": "snappy",
    },
    "models": {
        "clustering": {"n_clusters": 5, "algorithm": "kmeans", "auto_k": False},
        "forecasting": {"horizon_hours": 24, "confidence_interval": 0.95, "top_n_services": 10},
        "anomaly_detection": {"contamination": 0.10, "method": "isolation_forest"},
    },
    "reporting": {
        "output_dir": "reports/",
        "top_n_services": 15,
        "alert_thresholds": {"anomaly_score": 0.6, "growth_rate_pct": 30},
    },
    "logging": {"level": "INFO", "file": "logs/demo.log"},
}


# ---------------------------------------------------------------------------
# Run pipeline locally (bypassing AWS ingestion)
# ---------------------------------------------------------------------------

def run_demo():
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path(".cache_demo").mkdir(exist_ok=True)

    config = LOCAL_CONFIG

    from src.preprocessing.log_parser import LogParser
    from src.preprocessing.feature_engineering import FeatureEngineer
    from src.models.clustering import ServiceClusterer
    from src.models.forecasting import TrafficForecaster
    from src.models.anomaly_detection import AnomalyDetector
    from src.reporting.report_generator import ReportGenerator

    raw_df = make_synthetic_logs(hours=336)  # 14 days → enough for growth rate comparison

    parser = LogParser(config)
    parsed = parser.parse(raw_df)

    engineer = FeatureEngineer(config)
    service_features = engineer.build_service_features(parsed)
    ts_features = engineer.build_timeseries_features(parsed)
    growth_rates = engineer.compute_growth_rates(ts_features)

    clusterer = ServiceClusterer(config)
    clustered = clusterer.fit(service_features)

    forecaster = TrafficForecaster(config)
    forecasts = forecaster.forecast_all(ts_features)

    detector = AnomalyDetector(config)
    anomalies = detector.detect_service_anomalies(service_features)
    ts_anomalies = detector.detect_timeseries_anomalies(ts_features)

    reporter = ReportGenerator(config)
    report = reporter.generate(
        clustered_services=clustered,
        forecasts=forecasts,
        anomalies=anomalies,
        ts_anomalies=ts_anomalies,
        growth_rates=growth_rates,
        ts_features=ts_features,
        date_range={"start": "demo", "end": "demo"},
    )

    # Save report
    out = Path("reports/demo_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Full report saved → {out}")

    # Print highlights
    print("\n" + "=" * 68)
    print("  DEMO RESULTS")
    print("=" * 68)

    meta = report["metadata"]
    print(f"\n  Services analysed : {meta['total_services_analysed']}")
    print(f"  Total requests    : {meta['total_requests']:,}")

    print("\n  TOP 5 SERVICES BY TRAFFIC")
    print("  " + "-" * 50)
    for svc in report["traffic_summary"]["top_services"][:5]:
        print(
            f"  {svc['service_name']:<30} "
            f"{svc['requests_per_hour']:>8.1f} req/h  "
            f"err={svc['error_rate']*100:.1f}%  "
            f"cluster={svc['cluster_label']}"
        )

    print("\n  CLUSTER BREAKDOWN")
    print("  " + "-" * 50)
    for c in report["traffic_summary"]["cluster_breakdown"]:
        print(
            f"  {c['cluster_label']:<30} "
            f"{c['service_count']} services  "
            f"{c['total_requests']:>10,} reqs"
        )

    alerts = report["anomaly_alerts"]["service_alerts"]
    print(f"\n  ANOMALY ALERTS ({len(alerts)})")
    print("  " + "-" * 50)
    for a in alerts[:5]:
        print(f"  • {a['service']}  score={a['anomaly_score']:.3f}  → {a['recommendation']}")

    print("\n  COST OPTIMIZATION HIGHLIGHTS")
    print("  " + "-" * 50)
    for category, items in report["cost_optimization"].items():
        for item in items:
            print(f"  [{category}] {item['category']}: {item['estimated_saving']}")

    print("\n  Full report: reports/demo_report.json\n")


if __name__ == "__main__":
    run_demo()
