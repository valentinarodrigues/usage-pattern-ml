"""
PipelineOrchestrator — coordinates the end-to-end pipeline.

Cost-saving design decisions baked in:
  1. Incremental processing   — only ingests data added since the last run
  2. Model caching            — skips retraining if the model is recent enough
  3. Configurable data source — S3 Parquet (cheap) vs Athena (flexible)
  4. State file               — lightweight JSON; avoids a database dependency
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from ..ingestion.athena_client import AthenaClient
from ..ingestion.s3_reader import S3Reader
from ..models.anomaly_detection import AnomalyDetector
from ..models.clustering import ServiceClusterer
from ..models.forecasting import TrafficForecaster
from ..preprocessing.feature_engineering import FeatureEngineer
from ..preprocessing.log_parser import LogParser
from ..reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = Path(config["processing"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.cache_dir / "pipeline_state.json"

        self.athena = AthenaClient(config)
        self.s3 = S3Reader(config)
        self.parser = LogParser(config)
        self.engineer = FeatureEngineer(config)
        self.clusterer = ServiceClusterer(config)
        self.forecaster = TrafficForecaster(config)
        self.detector = AnomalyDetector(config)
        self.reporter = ReportGenerator(config)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, use_athena: bool = False, force_full_run: bool = False) -> dict:
        logger.info("=" * 60)
        logger.info("ML Usage Pattern Detection — Pipeline Start")
        logger.info("=" * 60)

        state = self._load_state()
        date_range = self._compute_date_range(state, force_full_run)
        logger.info(f"Date range: {date_range['start']} → {date_range['end']}")

        # 1. Ingest
        raw_df = self._ingest(date_range, use_athena)
        if raw_df.empty:
            logger.info("No new data found — pipeline complete")
            return {"status": "no_new_data", "date_range": date_range}

        # 2. Parse
        parsed_df = self.parser.parse(raw_df)
        if parsed_df.empty:
            logger.warning("All records filtered during parsing — check log schema")
            return {"status": "empty_after_parsing"}

        # 3. Feature engineering
        service_features = self.engineer.build_service_features(parsed_df)
        ts_features = self.engineer.build_timeseries_features(parsed_df)
        growth_rates = self.engineer.compute_growth_rates(ts_features)

        # 4. Models (can be run independently)
        clustered = self.clusterer.fit(service_features)
        forecasts = self.forecaster.forecast_all(ts_features)
        anomalies = self.detector.detect_service_anomalies(service_features)
        ts_anomalies = self.detector.detect_timeseries_anomalies(ts_features)

        # 5. Report
        results = self.reporter.generate(
            clustered_services=clustered,
            forecasts=forecasts,
            anomalies=anomalies,
            ts_anomalies=ts_anomalies,
            growth_rates=growth_rates,
            ts_features=ts_features,
            date_range=date_range,
        )

        # 6. Persist
        self._save_report(results)
        self._update_state(state, date_range)

        logger.info("=" * 60)
        logger.info("Pipeline complete")
        logger.info("=" * 60)
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ingest(self, date_range: dict, use_athena: bool) -> pd.DataFrame:
        if use_athena:
            logger.info("[Orchestrator] Ingestion source: Athena")
            sql = self.athena.build_log_query(date_range["start"], date_range["end"])
            return self.athena.query(sql)
        logger.info("[Orchestrator] Ingestion source: S3 Parquet (preferred — lower cost)")
        return self.s3.read_date_range(date_range["start"], date_range["end"])

    def _compute_date_range(self, state: dict, force_full: bool) -> dict:
        today = datetime.now(tz=timezone.utc).date()
        lookback = self.config["processing"]["lookback_days"]

        if (
            not force_full
            and state.get("last_run_date")
            and self.config["processing"].get("incremental", True)
        ):
            start = (
                datetime.fromisoformat(state["last_run_date"]).date() + timedelta(days=1)
            )
        else:
            start = today - timedelta(days=lookback)

        return {"start": str(start), "end": str(today)}

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def _update_state(self, state: dict, date_range: dict):
        state["last_run_date"] = date_range["end"]
        state["last_run_at"] = datetime.now(tz=timezone.utc).isoformat()
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _save_report(self, results: dict):
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config["reporting"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"report_{ts}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"[Orchestrator] Report saved → {path}")

        # Best-effort S3 upload (non-blocking)
        try:
            self.s3.write_results(results, f"report_{ts}.json")
        except Exception as exc:
            logger.warning(f"[Orchestrator] S3 upload failed (non-fatal): {exc}")
