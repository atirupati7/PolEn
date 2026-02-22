"""
FRED data client with local parquet caching.

Primary data source: Federal Reserve Economic Data (FRED) API.
Requires FRED_API_KEY environment variable for real data.
Falls back to synthetic data generation when key is missing.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)


class FREDClient:
    """Fetches and caches FRED economic data series."""

    def __init__(self):
        self.api_key = settings.fred_api_key
        self.raw_dir = settings.raw_cache_dir
        self.has_key = bool(self.api_key and self.api_key.strip())
        self._fred = None

    def _get_fred(self):
        """Lazy-init fredapi client."""
        if self._fred is None:
            if not self.has_key:
                raise ValueError("FRED_API_KEY not set. Use synthetic mode.")
            from fredapi import Fred
            self._fred = Fred(api_key=self.api_key)
        return self._fred

    def _cache_path(self, series_id: str) -> Path:
        return self.raw_dir / f"{series_id}.parquet"

    def _is_cache_fresh(self, path: Path, max_age_hours: int = 12) -> bool:
        """Check if cache file exists and is less than max_age_hours old."""
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return (datetime.now() - mtime) < timedelta(hours=max_age_hours)

    def fetch_series(self, series_id: str, force_refresh: bool = False) -> pd.Series:
        """
        Fetch a single FRED series. Uses parquet cache when fresh.

        Args:
            series_id: FRED series identifier (e.g., "SP500", "DGS10")
            force_refresh: If True, ignore cache and re-fetch

        Returns:
            pd.Series with DatetimeIndex
        """
        cache_path = self._cache_path(series_id)

        if not force_refresh and self._is_cache_fresh(cache_path):
            logger.info(f"Loading {series_id} from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0]

        if not self.has_key:
            raise ValueError(
                f"FRED_API_KEY not set. Cannot fetch {series_id}. Use synthetic mode."
            )

        logger.info(f"Fetching {series_id} from FRED API...")
        fred = self._get_fred()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * settings.data_lookback_years)

        try:
            data = fred.get_series(
                series_id,
                observation_start=start_date.strftime("%Y-%m-%d"),
                observation_end=end_date.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            # Try loading stale cache
            if cache_path.exists():
                logger.warning(f"Using stale cache for {series_id}")
                df = pd.read_parquet(cache_path)
                return df.iloc[:, 0]
            raise

        # Save to parquet
        df = pd.DataFrame({series_id: data})
        df.index.name = "date"
        df.to_parquet(cache_path)
        logger.info(f"Cached {series_id}: {len(df)} observations -> {cache_path}")

        return data

    def fetch_all(self, force_refresh: bool = False) -> dict[str, pd.Series]:
        """Fetch all configured FRED series."""
        results = {}
        errors = []
        for name, series_id in settings.fred_series.items():
            try:
                results[name] = self.fetch_series(series_id, force_refresh=force_refresh)
            except Exception as e:
                errors.append(f"{name} ({series_id}): {e}")
                logger.warning(f"Skipping {name}: {e}")

        if errors and not results:
            raise RuntimeError(
                f"Failed to fetch any FRED series:\n" + "\n".join(errors)
            )

        return results

    def get_cache_status(self) -> dict:
        """Return status of cached data files."""
        status = {}
        for name, series_id in settings.fred_series.items():
            path = self._cache_path(series_id)
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                df = pd.read_parquet(path)
                status[name] = {
                    "series_id": series_id,
                    "cached": True,
                    "cache_age_hours": round(
                        (datetime.now() - mtime).total_seconds() / 3600, 1
                    ),
                    "observations": len(df),
                    "start_date": str(df.index.min().date()) if len(df) > 0 else None,
                    "end_date": str(df.index.max().date()) if len(df) > 0 else None,
                }
            else:
                status[name] = {
                    "series_id": series_id,
                    "cached": False,
                }
        return status


def generate_synthetic_data(n_days: int = 9000) -> pd.DataFrame:
    """
    Generate realistic synthetic cross-asset data for demo mode.

    Produces ~36 years of daily data (1990–present) with realistic
    correlations, regime dynamics, inflation, and Fed rate series.

    Returns:
        DataFrame with columns matching FRED series names
    """
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.now(), periods=n_days, freq="B")

    # Generate correlated returns using a factor model
    n = len(dates)

    # Macro factors: growth, risk aversion, liquidity
    factors = np.random.randn(n, 3)
    # Add persistence
    for i in range(1, n):
        factors[i] = 0.97 * factors[i - 1] + 0.24 * factors[i]

    # SP500 level (start at ~1500, grow ~7% annualized with vol ~16%)
    sp500_returns = (
        0.07 / 252
        + 0.16 / np.sqrt(252) * (0.6 * factors[:, 0] - 0.3 * factors[:, 1] + 0.1 * np.random.randn(n))
    )
    sp500 = 1500 * np.exp(np.cumsum(sp500_returns))

    # Treasury yields with mean-reversion
    dgs10 = np.zeros(n)
    dgs2 = np.zeros(n)
    dgs5 = np.zeros(n)
    dgs10[0] = 3.5
    dgs2[0] = 2.5
    dgs5[0] = 3.0

    for i in range(1, n):
        dgs10[i] = dgs10[i - 1] + 0.01 * (3.5 - dgs10[i - 1]) + 0.05 * (
            0.2 * factors[i, 0] + 0.3 * factors[i, 2] + 0.5 * np.random.randn()
        )
        dgs2[i] = dgs2[i - 1] + 0.015 * (2.5 - dgs2[i - 1]) + 0.04 * (
            0.3 * factors[i, 0] + 0.4 * factors[i, 2] + 0.3 * np.random.randn()
        )
        dgs5[i] = dgs5[i - 1] + 0.012 * (3.0 - dgs5[i - 1]) + 0.045 * (
            0.25 * factors[i, 0] + 0.35 * factors[i, 2] + 0.4 * np.random.randn()
        )

    dgs10 = np.clip(dgs10, 0.1, 8.0)
    dgs2 = np.clip(dgs2, 0.05, 7.0)
    dgs5 = np.clip(dgs5, 0.07, 7.5)

    # BAA yield (spread over 10Y)
    baa_spread = np.zeros(n)
    baa_spread[0] = 2.0
    for i in range(1, n):
        baa_spread[i] = baa_spread[i - 1] + 0.02 * (2.0 - baa_spread[i - 1]) + 0.06 * (
            -0.2 * factors[i, 0] + 0.5 * factors[i, 1] + 0.3 * np.random.randn()
        )
    baa_spread = np.clip(baa_spread, 0.5, 6.0)
    baa = dgs10 + baa_spread

    # VIX (mean-reverting around 18)
    vix = np.zeros(n)
    vix[0] = 18.0
    for i in range(1, n):
        vix[i] = vix[i - 1] + 0.03 * (18.0 - vix[i - 1]) + 1.5 * (
            -0.3 * factors[i, 0] + 0.6 * factors[i, 1] + 0.1 * np.random.randn()
        )
    vix = np.clip(vix, 9.0, 80.0)

    # Dollar index (start ~100)
    usd_returns = (
        0.005 / 252
        + 0.08 / np.sqrt(252) * (0.1 * factors[:, 0] + 0.2 * factors[:, 1] + 0.7 * np.random.randn(n))
    )
    usd = 100 * np.exp(np.cumsum(usd_returns))

    # Introduce occasional regime shifts (crisis periods)
    crisis_periods = [
        (int(0.3 * n), int(0.32 * n)),   # ~2008 crisis
        (int(0.6 * n), int(0.62 * n)),   # ~2015 volatility
        (int(0.78 * n), int(0.80 * n)),  # ~2020 covid
    ]
    for start, end in crisis_periods:
        sp500[start:end] *= np.linspace(1.0, 0.75, end - start)
        vix[start:end] *= np.linspace(1.0, 2.5, end - start)
        baa_spread_boost = np.linspace(0, 2.0, end - start)
        baa[start:end] += baa_spread_boost

    # CPI level (monthly, interpolated to daily) — start ~130, grow ~2.5%/yr
    cpi = np.zeros(n)
    cpi[0] = 130.0  # CPI index level circa 1990
    for i in range(1, n):
        # ~2.5% annualized with some noise, plus factor sensitivity
        daily_infl = 0.025 / 252 + 0.001 * (0.2 * factors[i, 0] + 0.3 * np.random.randn())
        cpi[i] = cpi[i - 1] * (1 + daily_infl)
    # Introduce disinflation / inflation episodes
    for start, end in crisis_periods:
        # Deflationary pressure during crises
        deflation = np.linspace(1.0, 0.997, end - start)
        cpi[start:end] *= np.cumprod(deflation)

    # Federal Funds Rate (mean-reverting, responds to growth/inflation)
    fedfunds = np.zeros(n)
    fedfunds[0] = 5.0  # ~1990 level
    for i in range(1, n):
        # Taylor-rule-like mean reversion
        target = 2.5 + 0.5 * factors[i, 0]  # higher when growth is up
        fedfunds[i] = fedfunds[i - 1] + 0.005 * (target - fedfunds[i - 1]) + 0.02 * np.random.randn()
    fedfunds = np.clip(fedfunds, 0.0, 8.0)
    # ZLB episodes near crisis periods
    for start, end in crisis_periods:
        post_crisis_end = min(end + int(0.05 * n), n)
        fedfunds[start:post_crisis_end] *= np.linspace(1.0, 0.05, post_crisis_end - start)
        fedfunds[start:post_crisis_end] = np.clip(fedfunds[start:post_crisis_end], 0.0, 8.0)

    df = pd.DataFrame(
        {
            "SP500": sp500,
            "DGS2": dgs2,
            "DGS10": dgs10,
            "BAA": baa,
            "VIXCLS": vix,
            "DTWEXBGS": usd,
            "DGS5": dgs5,
            "CPIAUCSL": cpi,
            "FEDFUNDS": fedfunds,
        },
        index=dates,
    )
    df.index.name = "date"

    return df
