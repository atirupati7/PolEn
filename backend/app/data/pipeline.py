"""
Data preprocessing pipeline.

Aligns cross-asset data to business days, handles missing values,
computes transformations (returns, spreads, slopes), and applies
rolling z-score normalization.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.config import settings
from app.data.fred_client import FREDClient, generate_synthetic_data

logger = logging.getLogger(__name__)


class DataPipeline:
    """End-to-end data preprocessing from raw FRED series to model-ready features."""

    def __init__(self):
        self.client = FREDClient()
        self.processed_path = settings.processed_cache_dir / "merged.parquet"
        self._merged: Optional[pd.DataFrame] = None

    @property
    def is_synthetic(self) -> bool:
        return not self.client.has_key

    def refresh(self, force: bool = False, synthetic: bool = False) -> pd.DataFrame:
        """
        Full pipeline: fetch → align → transform → normalize → save.

        Args:
            force: Force re-fetch from FRED
            synthetic: Use synthetic data even if FRED key available

        Returns:
            Processed DataFrame ready for feature extraction
        """
        if synthetic or self.is_synthetic:
            logger.info("Using synthetic data (FRED_API_KEY not set or synthetic requested)")
            raw_df = generate_synthetic_data()
        else:
            raw_series = self.client.fetch_all(force_refresh=force)
            raw_df = self._merge_raw(raw_series)

        aligned = self._align_to_bdays(raw_df)
        transformed = self._compute_transforms(aligned)
        normalized = self._rolling_zscore(transformed)

        # Save processed
        normalized.to_parquet(self.processed_path)
        self._merged = normalized
        logger.info(f"Pipeline complete: {len(normalized)} days, {normalized.columns.tolist()}")
        return normalized

    def get_processed(self) -> pd.DataFrame:
        """Load processed data from cache or run pipeline."""
        if self._merged is not None:
            return self._merged
        if self.processed_path.exists():
            self._merged = pd.read_parquet(self.processed_path)
            return self._merged
        return self.refresh()

    def _merge_raw(self, series_dict: dict[str, pd.Series]) -> pd.DataFrame:
        """Merge individual series into a single DataFrame aligned by date."""
        frames = {}
        for name, s in series_dict.items():
            s = s.dropna()
            s.index = pd.DatetimeIndex(s.index)
            frames[name] = s

        df = pd.DataFrame(frames)
        df.index.name = "date"
        df = df.sort_index()
        return df

    def _align_to_bdays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align to business day calendar and handle missing values."""
        # Resample to business days
        bday_idx = pd.bdate_range(start=df.index.min(), end=df.index.max(), freq="B")
        df = df.reindex(bday_idx)
        df.index.name = "date"

        # Forward-fill daily yields for short gaps (<=3 days)
        daily_yield_cols = [c for c in ["DGS2", "DGS5", "DGS10"] if c in df.columns]
        for col in daily_yield_cols:
            df[col] = df[col].ffill(limit=3)

        # BAA corporate bond yield is monthly in FRED — ffill across all bdays
        if "BAA" in df.columns:
            df["BAA"] = df["BAA"].ffill()

        # VIX and dollar: ffill up to 3 days
        for col in ["VIXCLS", "DTWEXBGS"]:
            if col in df.columns:
                df[col] = df[col].ffill(limit=3)

        # SP500: do NOT ffill returns; but ffill level for up to 1 day for alignment
        if "SP500" in df.columns:
            df["SP500"] = df["SP500"].ffill(limit=1)

        # CPI: monthly series — ffill across all business days
        if "CPIAUCSL" in df.columns:
            df["CPIAUCSL"] = df["CPIAUCSL"].ffill()

        # Fed Funds Rate: monthly — ffill across business days
        if "FEDFUNDS" in df.columns:
            df["FEDFUNDS"] = df["FEDFUNDS"].ffill()

        # USREC: monthly recession indicator — ffill
        if "USREC" in df.columns:
            df["USREC"] = df["USREC"].ffill()

        # T10Y2Y: daily yield curve spread — ffill short gaps
        if "T10Y2Y" in df.columns:
            df["T10Y2Y"] = df["T10Y2Y"].ffill(limit=3)

        # Drop days where too many columns are missing (>50%)
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=int(threshold))

        return df

    def _compute_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features from raw levels.

        Transformations:
        - SP500: r_spx = log(SP500_t / SP500_{t-1})
        - DTWEXBGS: r_usd = log(USD_t / USD_{t-1})
        - VIXCLS: dvix = log(VIX_t / VIX_{t-1})  [log changes]
        - Rates: keep levels; add changes ΔDGS2, ΔDGS10
        - Credit spread: cs = BAA - DGS10
        - Curve slope: slope = DGS10 - DGS2
        - Curvature (if DGS5): curv = DGS10 - 2*DGS5 + DGS2
        """
        result = pd.DataFrame(index=df.index)

        # Log returns for SP500
        if "SP500" in df.columns:
            result["r_spx"] = np.log(df["SP500"] / df["SP500"].shift(1))

        # Log returns for dollar index
        if "DTWEXBGS" in df.columns:
            result["r_usd"] = np.log(df["DTWEXBGS"] / df["DTWEXBGS"].shift(1))

        # Log changes for VIX
        if "VIXCLS" in df.columns:
            result["dvix"] = np.log(df["VIXCLS"] / df["VIXCLS"].shift(1))

        # Yield levels
        for col in ["DGS2", "DGS10"]:
            if col in df.columns:
                result[col] = df[col]
                result[f"d_{col}"] = df[col].diff()

        # Credit spread
        if "BAA" in df.columns and "DGS10" in df.columns:
            result["cs"] = df["BAA"] - df["DGS10"]

        # Curve slope
        if "DGS10" in df.columns and "DGS2" in df.columns:
            result["slope"] = df["DGS10"] - df["DGS2"]

        # Curvature (optional)
        if all(c in df.columns for c in ["DGS10", "DGS5", "DGS2"]):
            result["curv"] = df["DGS10"] - 2 * df["DGS5"] + df["DGS2"]

        # VIX level for features
        if "VIXCLS" in df.columns:
            result["vix_level"] = df["VIXCLS"]

        # CPI: compute YoY inflation (uses ~252 business day lag for annual)
        if "CPIAUCSL" in df.columns:
            cpi = df["CPIAUCSL"]
            result["inflation_yoy"] = cpi.pct_change(periods=252)  # approx 1 year of bdays
            result["cpi_level"] = cpi

        # Federal Funds Rate level (as fraction)
        if "FEDFUNDS" in df.columns:
            result["fed_rate"] = df["FEDFUNDS"] / 100.0  # percent → fraction

        # NBER recession indicator (passthrough for backtest scoring)
        if "USREC" in df.columns:
            result["usrec"] = df["USREC"]

        # Yield curve spread (T10Y2Y) — inversion signal
        if "T10Y2Y" in df.columns:
            result["t10y2y"] = df["T10Y2Y"]

        # Drop first row (NaN from diff/returns)
        result = result.iloc[1:]

        # Drop rows with too many NaN critical columns.
        # We use a threshold rather than requiring ALL, because some series
        # (SP500 starts ~2013, DTWEXBGS ~2006) while others go back to 1990.
        critical = [c for c in ["r_spx", "dvix", "d_DGS10", "cs"] if c in result.columns]
        if critical:
            min_required = max(2, len(critical) - 1)  # allow 1 missing
            critical_present = result[critical].notnull().sum(axis=1)
            result = result[critical_present >= min_required]

        return result

    def _rolling_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling z-score normalization.

        Window: 252 business days (1 year)
        Min periods: 200

        z_t = (x_t - rolling_mean) / rolling_std
        """
        window = settings.rolling_zscore_window
        min_periods = settings.rolling_zscore_min_periods

        # Columns to z-score (the change/return columns, not levels or inflation)
        zscore_cols = [
            c for c in df.columns
            if (c.startswith("r_") or c.startswith("d_") or c.startswith("dvix") or c in ["cs", "slope", "curv"])
            and c not in ["inflation_yoy", "cpi_level", "fed_rate"]
        ]

        result = df.copy()

        for col in zscore_cols:
            roll_mean = df[col].rolling(window=window, min_periods=min_periods).mean()
            roll_std = df[col].rolling(window=window, min_periods=min_periods).std()
            # Avoid division by zero
            roll_std = roll_std.replace(0, np.nan)
            z_col = f"z_{col}"
            result[z_col] = (df[col] - roll_mean) / roll_std

        # Keep rows where a *majority* of z-scores are available, rather than
        # requiring ALL — otherwise the latest-starting series (z_r_spx from
        # ~2013) would truncate the entire dataset.
        z_cols = [c for c in result.columns if c.startswith("z_")]
        if z_cols:
            # Core z-score columns that should always be present
            core_z = [c for c in ["z_dvix", "z_d_DGS10", "z_cs"] if c in z_cols]
            if core_z:
                result = result.dropna(subset=core_z)
            else:
                result = result.dropna(subset=z_cols, how="all")

        return result
