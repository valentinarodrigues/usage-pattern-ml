"""
Normalizes raw log DataFrames into a consistent schema regardless of source format.
Handles flexible column names, type coercion, deduplication, and optional downsampling.
"""
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column names expected downstream
LOG_SCHEMA = [
    "timestamp",
    "service_name",
    "endpoint",
    "method",
    "status_code",
    "latency_ms",
    "bytes_response",
    "user_id",
    "instance_id",
]

# Maps common vendor column names → canonical names
COLUMN_ALIASES = {
    "ts": "timestamp",
    "time": "timestamp",
    "@timestamp": "timestamp",
    "event_time": "timestamp",
    "svc": "service_name",
    "service": "service_name",
    "app": "service_name",
    "application": "service_name",
    "path": "endpoint",
    "url": "endpoint",
    "request_path": "endpoint",
    "uri": "endpoint",
    "http_method": "method",
    "request_method": "method",
    "verb": "method",
    "status": "status_code",
    "http_status": "status_code",
    "response_code": "status_code",
    "http_status_code": "status_code",
    "latency": "latency_ms",
    "response_time": "latency_ms",
    "duration_ms": "latency_ms",
    "elapsed_ms": "latency_ms",
    "response_size": "bytes_response",
    "content_length": "bytes_response",
    "bytes_sent": "bytes_response",
    "user": "user_id",
    "userid": "user_id",
    "client_id": "user_id",
    "instance": "instance_id",
    "host": "instance_id",
    "pod": "instance_id",
    "server": "instance_id",
}

REQUIRED_COLUMNS = ["timestamp", "service_name", "status_code"]

# Regex to normalise dynamic path segments (e.g. /users/123 → /users/{id})
_ID_PATTERN = re.compile(r"/\d{2,}")
_UUID_PATTERN = re.compile(
    r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)


class LogParser:
    def __init__(self, config: dict):
        self.sample_rate: float = config["processing"].get("sample_rate", 1.0)

    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"[Parser] Input: {len(df):,} records")
        df = (
            df.pipe(self._rename_columns)
            .pipe(self._add_missing_columns)
            .pipe(self._validate_required)
            .pipe(self._coerce_types)
            .pipe(self._clean_values)
            .pipe(self._normalize_endpoints)
            .pipe(self._deduplicate)
            .pipe(self._downsample)
        )
        logger.info(f"[Parser] Output: {len(df):,} clean records")
        return df

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
        return df.rename(columns=mapping)

    def _add_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in LOG_SCHEMA:
            if col not in df.columns:
                df[col] = np.nan if col in ("latency_ms", "bytes_response") else "unknown"
        return df

    def _validate_required(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Required columns missing after renaming: {missing}")
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["status_code"] = pd.to_numeric(df["status_code"], errors="coerce").astype("Int16")
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce").astype("float32")
        df["bytes_response"] = pd.to_numeric(df["bytes_response"], errors="coerce").astype("float32")
        df["service_name"] = df["service_name"].astype(str).str.strip().str.lower()
        df["endpoint"] = df["endpoint"].astype(str).str.strip()
        df["method"] = df["method"].astype(str).str.upper().str.strip()
        return df

    def _clean_values(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["timestamp", "service_name", "status_code"])
        df = df[df["status_code"].between(100, 599)]
        # Latency sanity check: drop negatives and anything >5 minutes (300 000 ms)
        valid_latency = df["latency_ms"].isna() | df["latency_ms"].between(0, 300_000)
        df = df[valid_latency]
        dropped = before - len(df)
        if dropped:
            logger.info(f"[Parser] Dropped {dropped:,} invalid records ({dropped/before*100:.1f}%)")
        return df

    def _normalize_endpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip query strings, then replace numeric IDs with {id} placeholders
        df["endpoint"] = (
            df["endpoint"]
            .str.split("?").str[0]
            .str.replace(_UUID_PATTERN, "/{uuid}", regex=True)
            .str.replace(_ID_PATTERN, "/{id}", regex=True)
        )
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        key_cols = [c for c in ["timestamp", "service_name", "endpoint", "instance_id"] if c in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=key_cols)
        removed = before - len(df)
        if removed:
            logger.info(f"[Parser] Removed {removed:,} duplicates")
        return df.reset_index(drop=True)

    def _downsample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sample_rate < 1.0:
            df = df.sample(frac=self.sample_rate, random_state=42).reset_index(drop=True)
            logger.info(f"[Parser] Downsampled to {len(df):,} rows (rate={self.sample_rate})")
        return df
