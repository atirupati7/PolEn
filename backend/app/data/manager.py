"""
DataManager — unified data orchestrator for the MacroState system.

Handles:
  - download_data()        : fetch from FRED or generate synthetic
  - load_local_data()      : load from parquet cache
  - preprocess()           : transforms + z-score normalisation
  - align_frequency()      : daily → monthly aggregation
  - build_state_matrix()   : Kalman (3-dim) + inflation gap → 4-dim state
  - get_inflation_series() : CPI YoY → inflation gap time-series
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)


class DataManager:
    """
    Central data orchestrator.

    Stores both daily and monthly processed frames and exposes them
    to downstream modules (Kalman, RL env, Monte Carlo, API).
    """

    def __init__(self) -> None:
        from app.data.fred_client import FREDClient
        from app.data.pipeline import DataPipeline

        self.client = FREDClient()
        self.pipeline = DataPipeline()

        self._daily_processed: Optional[pd.DataFrame] = None
        self._monthly_processed: Optional[pd.DataFrame] = None
        self._inflation_series: Optional[pd.Series] = None
        self._fed_rate_series: Optional[pd.Series] = None

    # ── public API ──────────────────────────────────────────────

    def download_data(self, force: bool = False, synthetic: bool = False) -> pd.DataFrame:
        """Download / generate daily data and run full pipeline."""
        df = self.pipeline.refresh(force=force, synthetic=synthetic)
        self._daily_processed = df
        self._build_monthly(df)
        self._extract_inflation()
        self._extract_fed_rate()
        return df

    def load_local_data(self) -> pd.DataFrame:
        """Load from cache or run pipeline."""
        df = self.pipeline.get_processed()
        self._daily_processed = df
        self._build_monthly(df)
        self._extract_inflation()
        self._extract_fed_rate()
        return df

    def preprocess(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Re-run preprocessing on provided or cached daily data."""
        if df is None:
            df = self._daily_processed
        if df is None:
            return self.download_data()
        return df

    @property
    def daily(self) -> Optional[pd.DataFrame]:
        return self._daily_processed

    @property
    def monthly(self) -> Optional[pd.DataFrame]:
        return self._monthly_processed

    def align_frequency(self, freq: str = "monthly") -> pd.DataFrame:
        """Return data at the requested frequency."""
        if freq == "daily":
            if self._daily_processed is None:
                self.load_local_data()
            return self._daily_processed  # type: ignore[return-value]
        # Monthly
        if self._monthly_processed is None:
            if self._daily_processed is None:
                self.load_local_data()
            self._build_monthly(self._daily_processed)
        return self._monthly_processed  # type: ignore[return-value]

    def get_inflation_series(self) -> pd.Series:
        """
        Return year-over-year CPI inflation as a monthly time-series.

        Falls back to synthetic if CPI data not available.
        """
        if self._inflation_series is not None:
            return self._inflation_series
        self._extract_inflation()
        if self._inflation_series is None:
            # Synthetic fallback
            n = 400
            dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="M")
            rng = np.random.default_rng(42)
            inflation = 0.02 + 0.005 * np.cumsum(rng.normal(0, 0.1, n))
            inflation = np.clip(inflation, -0.02, 0.10)
            self._inflation_series = pd.Series(inflation, index=dates, name="inflation_yoy")
        return self._inflation_series

    def get_fed_rate_series(self) -> pd.Series:
        """Return the federal funds rate as a monthly series."""
        if self._fed_rate_series is not None:
            return self._fed_rate_series
        self._extract_fed_rate()
        if self._fed_rate_series is None:
            # Synthetic fallback
            n = 400
            dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="M")
            rng = np.random.default_rng(42)
            rate = np.zeros(n)
            rate[0] = 3.0
            for i in range(1, n):
                rate[i] = rate[i - 1] + 0.01 * (3.0 - rate[i - 1]) + 0.1 * rng.normal()
            rate = np.clip(rate, 0.0, 8.0)
            self._fed_rate_series = pd.Series(rate, index=dates, name="fed_rate")
        return self._fed_rate_series

    def get_inflation_gap(self) -> pd.Series:
        """inflation_gap = inflation_yoy - inflation_target."""
        inflation = self.get_inflation_series()
        return inflation - settings.inflation_target

    def build_state_matrix(
        self,
        kalman_states: np.ndarray,
        kalman_dates: Optional[pd.DatetimeIndex] = None,
    ) -> np.ndarray:
        """
        Combine Kalman 3-dim latent states with inflation gap → 4-dim state.

        Args:
            kalman_states: (T, 3) array [stress, liquidity, growth]
            kalman_dates:  DatetimeIndex aligned to kalman_states (optional)

        Returns:
            (T, 4) array [stress, liquidity, growth, inflation_gap]
        """
        T = kalman_states.shape[0]
        inflation_gap = self.get_inflation_gap()

        if kalman_dates is not None and len(inflation_gap) > 0:
            # Align inflation gap to kalman dates (monthly → resample if needed)
            infl_reindexed = inflation_gap.reindex(kalman_dates, method="ffill")
            infl_vals = infl_reindexed.values[-T:]
        else:
            # Use last T values of inflation gap
            infl_vals = inflation_gap.values[-T:] if len(inflation_gap) >= T else np.zeros(T)
            if len(infl_vals) < T:
                infl_vals = np.pad(infl_vals, (T - len(infl_vals), 0), mode="edge")

        # Clip for stability
        infl_vals = np.clip(infl_vals, -0.10, 0.10).astype(np.float32)

        state_4d = np.column_stack([kalman_states, infl_vals[:T]])
        return state_4d

    # ── private helpers ─────────────────────────────────────────

    def _build_monthly(self, df: Optional[pd.DataFrame]) -> None:
        """Aggregate daily processed data to monthly (last bday of month)."""
        if df is None or len(df) == 0:
            return

        # For z-scored columns take mean, for levels take last
        z_cols = [c for c in df.columns if c.startswith("z_")]
        level_cols = [c for c in df.columns if not c.startswith("z_")]

        monthly_parts: list[pd.DataFrame] = []
        if z_cols:
            monthly_parts.append(df[z_cols].resample("M").mean())
        if level_cols:
            monthly_parts.append(df[level_cols].resample("M").last())

        if monthly_parts:
            self._monthly_processed = pd.concat(monthly_parts, axis=1).dropna(how="all")
        logger.info(
            f"Monthly data built: {len(self._monthly_processed)} months"
            if self._monthly_processed is not None
            else "Monthly build failed"
        )

    def _extract_inflation(self) -> None:
        """Extract CPI YoY inflation from raw FRED cache."""
        try:
            cpi_path = settings.raw_cache_dir / "CPIAUCSL.parquet"
            if cpi_path.exists():
                cpi_df = pd.read_parquet(cpi_path)
                cpi = cpi_df.iloc[:, 0].dropna()
                cpi.index = pd.DatetimeIndex(cpi.index)
                cpi = cpi.sort_index()
                # YoY inflation
                cpi_yoy = cpi.pct_change(periods=12)
                self._inflation_series = cpi_yoy.dropna()
                self._inflation_series.name = "inflation_yoy"
                logger.info(f"CPI inflation extracted: {len(self._inflation_series)} obs")
            else:
                logger.info("No CPI cache found — will use synthetic inflation")
        except Exception as e:
            logger.warning(f"Failed to extract CPI inflation: {e}")

    def _extract_fed_rate(self) -> None:
        """Extract Federal Funds Rate from raw FRED cache."""
        try:
            ff_path = settings.raw_cache_dir / "FEDFUNDS.parquet"
            if ff_path.exists():
                ff_df = pd.read_parquet(ff_path)
                ff = ff_df.iloc[:, 0].dropna()
                ff.index = pd.DatetimeIndex(ff.index)
                ff = ff.sort_index()
                # Convert from percent to fraction for consistency
                self._fed_rate_series = ff / 100.0
                self._fed_rate_series.name = "fed_rate"
                logger.info(f"Fed funds rate extracted: {len(self._fed_rate_series)} obs")
            else:
                logger.info("No FEDFUNDS cache found — will use synthetic rate")
        except Exception as e:
            logger.warning(f"Failed to extract fed funds rate: {e}")
