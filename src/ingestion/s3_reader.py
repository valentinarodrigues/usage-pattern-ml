"""
Cost-optimized S3 reader:
  - Parquet format        → columnar storage, ~75% smaller than CSV, 80% cheaper Athena queries
  - Column pruning        → read only needed columns, reducing I/O
  - Partition-aware reads → skip irrelevant date partitions entirely
  - Incremental tracking  → remember last-processed key, skip re-reads
"""
import io
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class S3Reader:
    def __init__(self, config: dict):
        self.region = config["aws"]["region"]
        self.bucket = config["aws"]["s3"]["logs_bucket"]
        self.logs_prefix = config["aws"]["s3"]["logs_prefix"]
        self.parquet_prefix = config["aws"]["s3"]["parquet_prefix"]
        self.results_prefix = config["aws"]["s3"]["results_prefix"]

        self.cache_dir = Path(config["processing"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.cache_dir / "s3_incremental_state.json"

        self.s3 = boto3.client("s3", region_name=self.region)

    # ------------------------------------------------------------------
    # Primary read interface
    # ------------------------------------------------------------------

    def read_date_range(
        self,
        start_date: str,
        end_date: str,
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Read partitioned Parquet files for a date range.
        Assumes Hive-style partitioning: prefix/year=YYYY/month=MM/day=DD/
        Only touches partitions within the requested range — skips everything else.
        """
        keys = self._list_partitioned_keys(start_date, end_date)
        if not keys:
            logger.info("[S3] No Parquet files found for range — returning empty DataFrame")
            return pd.DataFrame()

        logger.info(f"[S3] Reading {len(keys)} Parquet files ({start_date} → {end_date})")
        return self._read_parquet_files(keys, columns=columns)

    def read_incremental(self, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Read only files not yet processed (tracked via state file).
        Ideal for daily pipeline runs — avoids re-reading historical data.
        """
        state = self._load_state()
        last_key = state.get("last_processed_key")

        new_keys = self._list_new_files(self.parquet_prefix, after_key=last_key)
        if not new_keys:
            logger.info("[S3] No new files since last run")
            return pd.DataFrame()

        logger.info(f"[S3] {len(new_keys)} new files to process")
        df = self._read_parquet_files(new_keys, columns=columns)

        state["last_processed_key"] = new_keys[-1]
        self._save_state(state)
        return df

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def write_parquet(self, df: pd.DataFrame, key: str, compression: str = "snappy"):
        """
        Write DataFrame as Parquet to S3.
        snappy: fast compression, ~60% size reduction vs CSV — good for query performance.
        gzip:  better compression (~75%), slower — better for archival.
        """
        buffer = io.BytesIO()
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, buffer, compression=compression)
        buffer.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.read())
        logger.info(f"[S3] Wrote {len(df):,} rows → s3://{self.bucket}/{key}")

    def write_results(self, data: dict, filename: str):
        key = f"{self.results_prefix}{filename}"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data, indent=2, default=str),
            ContentType="application/json",
        )
        logger.info(f"[S3] Results written → s3://{self.bucket}/{key}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _list_partitioned_keys(self, start_date: str, end_date: str) -> list:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        keys = []
        current = start
        while current <= end:
            prefix = (
                f"{self.parquet_prefix}"
                f"year={current.year}/"
                f"month={current.month:02d}/"
                f"day={current.day:02d}/"
            )
            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".parquet"):
                        keys.append(obj["Key"])
            current += timedelta(days=1)
        return sorted(keys)

    def _list_new_files(self, prefix: str, after_key: Optional[str] = None) -> list:
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith(".parquet") and (after_key is None or k > after_key):
                    keys.append(k)
        return sorted(keys)

    def _read_parquet_files(self, keys: list, columns: Optional[list] = None) -> pd.DataFrame:
        dfs = []
        for key in keys:
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=key)
                buf = pa.BufferReader(response["Body"].read())
                table = pq.read_table(buf, columns=columns)
                dfs.append(table.to_pandas())
            except Exception as exc:
                logger.warning(f"[S3] Skipped {key}: {exc}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def _save_state(self, state: dict):
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
