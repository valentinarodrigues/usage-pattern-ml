"""
Athena client with built-in cost controls:
  - Query result caching  → avoids re-running identical queries (saves $$)
  - Max bytes scanned limit → hard stop on runaway queries
  - Partition-pruned query builder → minimize data scanned
  - Workgroup enforcement → centralised cost governance
"""
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


class AthenaClient:
    def __init__(self, config: dict):
        self.region = config["aws"]["region"]
        self.database = config["aws"]["athena"]["database"]
        self.workgroup = config["aws"]["athena"]["workgroup"]
        self.results_bucket = config["aws"]["athena"]["query_results_bucket"]
        self.max_scan_mb = config["aws"]["athena"].get("max_scan_mb", 1000)
        self.table = config["aws"]["athena"].get("table", "access_logs")

        self.cache_dir = Path(config["processing"]["cache_dir"]) / "athena"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = boto3.client("athena", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, sql: str, use_cache: bool = True) -> pd.DataFrame:
        """Execute SQL and return DataFrame. Caches results locally to avoid re-billing."""
        query_hash = self._hash(sql)
        cache_path = self.cache_dir / f"{query_hash}.parquet"

        if use_cache and cache_path.exists():
            logger.info(f"[Athena] Cache hit {query_hash} — skipped re-scan cost")
            return pd.read_parquet(cache_path)

        logger.info(f"[Athena] Executing query {query_hash}")
        execution_id = self._start_query(sql)
        result_meta = self._poll(execution_id)
        self._enforce_scan_limit(result_meta)

        output_loc = result_meta["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
        df = self._download_csv(output_loc)

        df.to_parquet(cache_path, compression="snappy", index=False)
        logger.info(f"[Athena] Cached {len(df)} rows → {cache_path}")
        return df

    def build_log_query(self, start_date: str, end_date: str) -> str:
        """
        Partition-pruned query.
        The WHERE dt BETWEEN clause is key — without it Athena scans the full table.
        """
        return f"""
            SELECT
                dt,
                service_name,
                endpoint,
                method,
                status_code,
                latency_ms,
                bytes_response,
                user_id,
                instance_id
            FROM {self.database}.{self.table}
            WHERE dt BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY dt
        """

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _start_query(self, sql: str) -> str:
        response = self.client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.results_bucket},
            WorkGroup=self.workgroup,
        )
        return response["QueryExecutionId"]

    def _poll(self, execution_id: str, poll_interval: float = 2.0) -> dict:
        while True:
            result = self.client.get_query_execution(QueryExecutionId=execution_id)
            state = result["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                return result
            if state in ("FAILED", "CANCELLED"):
                reason = result["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
                raise RuntimeError(f"Athena query {state}: {reason}")
            time.sleep(poll_interval)

    def _enforce_scan_limit(self, result: dict):
        stats = result["QueryExecution"].get("Statistics", {})
        scanned_bytes = stats.get("DataScannedInBytes", 0)
        scanned_mb = scanned_bytes / (1024 * 1024)
        cost_usd = scanned_mb / 1024 * 5  # $5 per TB
        logger.info(f"[Athena] Scanned {scanned_mb:.1f} MB  (est. ${cost_usd:.4f})")
        if scanned_mb > self.max_scan_mb:
            raise RuntimeError(
                f"Query scanned {scanned_mb:.0f} MB, limit is {self.max_scan_mb} MB. "
                "Add partition filters to reduce cost."
            )

    def _download_csv(self, output_location: str) -> pd.DataFrame:
        bucket, key = output_location.replace("s3://", "").split("/", 1)
        response = self.s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(response["Body"])

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]
