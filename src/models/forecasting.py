"""
Hourly traffic forecasting per service using Exponential Smoothing (ETS).

Why ETS over SARIMA by default:
  - ~10× faster to fit (lower compute cost on many services)
  - Handles daily seasonality well
  - No stationarity pre-processing required

SARIMA is available as an alternative for services that need explicit AR/MA terms.
"""
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class TrafficForecaster:
    def __init__(self, config: dict):
        cfg = config["models"]["forecasting"]
        self.horizon: int = cfg.get("horizon_hours", 24)
        self.ci_z: float = 1.96 if cfg.get("confidence_interval", 0.95) == 0.95 else 2.576
        self.top_n: int = cfg.get("top_n_services", 20)

        self.cache_dir = Path(config["processing"]["cache_dir"]) / "models" / "forecasts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast_all(self, ts: pd.DataFrame) -> Dict[str, Optional[pd.DataFrame]]:
        """Forecast the top-N services by total traffic volume."""
        top_services = (
            ts.groupby("service_name")["request_count"]
            .sum()
            .nlargest(self.top_n)
            .index.tolist()
        )

        results: Dict[str, Optional[pd.DataFrame]] = {}
        for service in top_services:
            try:
                svc_ts = ts[ts["service_name"] == service].sort_values("time_bucket")
                results[service] = self._forecast_one(service, svc_ts)
            except Exception as exc:
                logger.warning(f"[Forecast] {service} failed: {exc}")
                results[service] = None

        succeeded = sum(1 for v in results.values() if v is not None)
        logger.info(f"[Forecast] {succeeded}/{len(top_services)} services forecasted")
        return results

    def identify_peak_windows(self, forecasts: Dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
        rows = []
        for service, fc in forecasts.items():
            if fc is None or fc.empty:
                continue
            idx = fc["forecast"].idxmax()
            row = fc.loc[idx]
            rows.append(
                {
                    "service_name": service,
                    "peak_time": row["time_bucket"],
                    "peak_forecast": round(float(row["forecast"]), 1),
                    "peak_upper_bound": round(float(row["upper_bound"]), 1),
                }
            )
        return pd.DataFrame(rows).sort_values("peak_forecast", ascending=False)

    def compute_capacity_headroom(
        self,
        forecasts: Dict[str, Optional[pd.DataFrame]],
        current_capacity: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Flags under/over-provisioned services.
        Uses upper_bound (not mean) as the safety target — better for capacity planning.
        """
        rows = []
        for service, fc in forecasts.items():
            if fc is None or fc.empty:
                continue
            peak = float(fc["upper_bound"].max())
            current = (current_capacity or {}).get(service, peak * 1.5)
            headroom_pct = (current - peak) / current * 100 if current > 0 else 0
            rows.append(
                {
                    "service_name": service,
                    "peak_forecast_upper": round(peak, 1),
                    "current_capacity": round(current, 1),
                    "headroom_pct": round(headroom_pct, 1),
                    "status": (
                        "under_provisioned"
                        if headroom_pct < 10
                        else "over_provisioned"
                        if headroom_pct > 50
                        else "well_provisioned"
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("headroom_pct")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _forecast_one(self, service: str, svc_ts: pd.DataFrame) -> Optional[pd.DataFrame]:
        series = (
            svc_ts.set_index("time_bucket")["request_count"]
            .asfreq("h", fill_value=0)
            .astype(float)
        )

        if len(series) < 48:
            logger.debug(f"[Forecast] {service}: only {len(series)}h of data — skipping")
            return None

        # Exponential Smoothing with additive daily seasonality
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=24,
            damped_trend=True,
        )
        fit = model.fit(optimized=True, use_brute=False)

        forecast_mean = fit.forecast(self.horizon).clip(lower=0)
        resid_std = float(np.std(fit.resid))

        fc_df = pd.DataFrame(
            {
                "time_bucket": forecast_mean.index,
                "forecast": forecast_mean.values.round(2),
                "lower_bound": (forecast_mean - self.ci_z * resid_std).clip(lower=0).values.round(2),
                "upper_bound": (forecast_mean + self.ci_z * resid_std).values.round(2),
            }
        )

        # Cache the fitted model for later inspection
        safe_name = service.replace("/", "_").replace(":", "_")
        joblib.dump(fit, self.cache_dir / f"{safe_name}.pkl")
        return fc_df
