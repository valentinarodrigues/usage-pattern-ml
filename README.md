# ML-Based Usage Pattern Detection from Logs

Detects trends, peak loads, and high-traffic services from structured access logs using machine learning. Outputs actionable capacity recommendations and cost optimisation insights.

**Tech stack:** Python · Pandas · Scikit-learn · Statsmodels · AWS Athena · S3 · Glue

---

## What it does

| Stage | What happens |
|---|---|
| **Ingest** | Pulls logs from S3 Parquet (cheap) or Athena (flexible), incrementally |
| **Parse** | Normalises raw logs into a consistent schema regardless of source format |
| **Features** | Builds per-service aggregates + hourly time-series grids |
| **Cluster** | Groups services by usage profile (KMeans / DBSCAN) |
| **Forecast** | Predicts traffic 24h ahead per service with confidence intervals |
| **Anomaly** | Flags unusual services and time windows (Isolation Forest + Z-score) |
| **Report** | Outputs structured JSON with alerts, scale recommendations, and cost savings |

---

## Project structure

```
usage-pattern-ml/
├── config/
│   └── config.yaml                 # All tunable parameters
├── src/
│   ├── ingestion/
│   │   ├── athena_client.py        # Partition-pruned queries + result caching
│   │   └── s3_reader.py            # Parquet reads + incremental state tracking
│   ├── preprocessing/
│   │   ├── log_parser.py           # Schema normalisation, dedup, downsampling
│   │   └── feature_engineering.py  # Service features + hourly time series
│   ├── models/
│   │   ├── clustering.py           # KMeans / DBSCAN service segmentation
│   │   ├── forecasting.py          # Exponential Smoothing 24h forecasting
│   │   └── anomaly_detection.py    # Isolation Forest + Z-score detection
│   ├── pipeline/
│   │   └── orchestrator.py         # Incremental pipeline coordinator
│   └── reporting/
│       └── report_generator.py     # Insights + cost recommendations
├── main.py                         # CLI entry point (production, requires AWS)
├── demo_local.py                   # Offline demo on synthetic data (no AWS needed)
└── requirements.txt
```

---

## Run locally (no AWS required)

The demo generates 3.6M synthetic log records across 15 services and runs the full pipeline locally.

### 1. Clone the repo

```bash
git clone https://github.com/valentinarodrigues/usage-pattern-ml.git
cd usage-pattern-ml
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the demo

```bash
python demo_local.py
```

You will see output like:

```
  Services analysed : 15
  Total requests    : 3,624,520

  TOP 5 SERVICES BY TRAFFIC
  content-delivery          3787.8 req/h  err=0.1%  cluster=high_traffic_stable
  product-catalog           2324.7 req/h  err=0.5%  cluster=high_traffic_stable
  ...

  ANOMALY ALERTS (1)
  • legacy-reports  score=1.000  → High error rate — investigate health and dependencies

  COST OPTIMIZATION HIGHLIGHTS
  [aws_compute] Spot Instance Candidates: HIGH — Spot is 60–90% cheaper than On-Demand
  [aws_storage] Log Format: CSV → Parquet: HIGH — ~75% storage reduction; ~80% Athena cost reduction
  ...
```

The full structured report is saved to `reports/demo_report.json`.

---

## Run in production (AWS)

### Prerequisites

- AWS credentials configured (`aws configure` or environment variables)
- S3 bucket with logs in Parquet format, partitioned by `year=/month=/day=/`
- Athena database and table registered in AWS Glue Catalog

### 1. Configure

Edit `config/config.yaml`:

```yaml
aws:
  region: us-east-1
  s3:
    logs_bucket: your-logs-bucket      # ← your bucket
    parquet_prefix: logs/parquet/
  athena:
    database: logs_db                  # ← your Glue database
    workgroup: ml-usage-analysis
    query_results_bucket: s3://your-athena-results/
    max_scan_mb: 1000                  # hard stop on runaway queries
```

### 2. Run

```bash
# Incremental run (S3 Parquet — recommended, lowest cost)
python main.py

# Use Athena as the data source
python main.py --athena

# Reprocess full history (ignore incremental state)
python main.py --full-run

# Custom config file
python main.py --config path/to/config.yaml
```

---

## Cost optimisation features built in

| Technique | Saving |
|---|---|
| S3 Parquet + Snappy (vs CSV) | ~75% storage, ~80% Athena query cost |
| S3 Hive partitioning + partition pruning | 70–95% per-query scan reduction |
| Athena result caching (local + workgroup) | Eliminates repeat scan charges |
| Incremental pipeline (delta-only runs) | 95%+ daily scan volume reduction |
| Spot Instance candidate detection | 60–90% compute saving on flagged services |
| Idle service detection (<1 req/h) | Eliminates always-on compute for dead services |
| S3 Lifecycle policy recommendations | 40–80% saving on logs older than 90 days |
| Off-peak scheduling guidance | 15–30% lower Spot prices at 02–05 UTC |
| Model caching (joblib) | Skips retraining on unchanged data |
| Downsampling for ML training | 70–90% faster feature engineering |

---

## Configuration reference

| Key | Default | Description |
|---|---|---|
| `processing.lookback_days` | `30` | History window for full runs |
| `processing.incremental` | `true` | Only process new data since last run |
| `processing.sample_rate` | `1.0` | Reduce to `0.1`–`0.3` for large datasets |
| `models.clustering.algorithm` | `kmeans` | `kmeans` or `dbscan` |
| `models.clustering.n_clusters` | `8` | Number of service segments |
| `models.clustering.auto_k` | `false` | Auto-select K via silhouette score |
| `models.forecasting.horizon_hours` | `24` | Forecast horizon |
| `models.forecasting.top_n_services` | `20` | Limit forecasting to top N services |
| `models.anomaly_detection.contamination` | `0.05` | Expected anomaly fraction |
| `models.anomaly_detection.method` | `isolation_forest` | `isolation_forest` or `zscore` |
| `reporting.alert_thresholds.growth_rate_pct` | `50` | Alert threshold for WoW growth |

---

## Output

Reports are saved to `reports/report_<timestamp>.json` and optionally uploaded to S3. The report contains:

```json
{
  "metadata":        { "total_services_analysed": 15, "total_requests": 3624520 },
  "traffic_summary": { "top_services": [...], "cluster_breakdown": [...], "fast_growing_services": [...] },
  "anomaly_alerts":  { "service_alerts": [...], "time_window_alerts": [...] },
  "capacity_plan":   { "forecast_peak_windows": [...], "scale_recommendations": [...] },
  "cost_optimization": {
    "aws_compute":      [...],
    "aws_storage":      [...],
    "data_processing":  [...],
    "architectural":    [...]
  }
}
```
