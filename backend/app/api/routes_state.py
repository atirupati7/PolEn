"""
API routes for state management and data status.
"""

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["state"])

# Shared state store (populated by refresh)
_state_store: dict = {}


def get_state_store() -> dict:
    return _state_store


@router.get("/health")
async def health():
    return {"ok": True}


@router.get("/data/status")
async def data_status():
    """Return status of cached data."""
    from app.data.fred_client import FREDClient
    client = FREDClient()
    status = client.get_cache_status()
    return {
        "has_fred_key": client.has_key,
        "series": status,
    }


@router.post("/state/refresh")
async def refresh_state(synthetic: bool = False):
    """
    Refresh data, preprocess, compute structure features, and run Kalman+EM.
    """
    global _state_store
    try:
        from app.data.manager import DataManager
        from app.models.structure_features import compute_structure_features
        from app.models.kalman_em import KalmanEM

        # Step 1: Run data pipeline via DataManager
        dm = DataManager()
        use_synthetic = synthetic or dm.pipeline.is_synthetic
        if use_synthetic:
            df = dm.download_data(synthetic=True)
        else:
            df = dm.download_data(force=True)

        logger.info(f"Data pipeline complete: {len(df)} rows, cols={df.columns.tolist()}")

        # Step 2: Compute structure features
        struct = compute_structure_features(df, window=settings.rolling_cov_window)

        logger.info(f"Structure features computed: Z_t dim={len(struct['Z_t'])}")

        # Step 3: Run Kalman + EM
        Z_history = struct["Z_history"].values
        kalman = KalmanEM(latent_dim=settings.kalman_latent_dim)
        fit_result = kalman.fit(Z_history)

        # Step 4: Get latest state
        latest_state = kalman.get_latest_state()

        # Step 5: Build 4-dim state with inflation
        smoothed_4d = dm.build_state_matrix(kalman.smoothed_means)
        inflation_gap = dm.get_inflation_gap()
        fed_rate = dm.get_fed_rate_series()

        # Store everything
        _state_store = {
            "df": df,
            "structure": struct,
            "kalman": kalman,
            "data_manager": dm,
            "fit_result": fit_result,
            "latest_state": latest_state,
            "smoothed_4d": smoothed_4d,
            "inflation_gap": float(inflation_gap.iloc[-1]) if len(inflation_gap) > 0 else 0.0,
            "fed_rate": float(fed_rate.iloc[-1]) if len(fed_rate) > 0 else 0.03,
            "latest_date": str(df.index[-1].date()),
            "is_synthetic": use_synthetic,
        }

        # Step 6: Build historical snapshots for timeline browsing
        _state_store["historical_snapshots"] = _build_historical_snapshots(
            kalman=kalman,
            struct=struct,
            dm=dm,
        )
        logger.info(
            f"Built {len(_state_store['historical_snapshots'])} historical snapshots"
        )

        return {
            "status": "ok",
            "latest_date": _state_store["latest_date"],
            "regime_label": latest_state["regime_label"],
            "stress_score": latest_state["stress_score"],
            "inflation_gap": _state_store["inflation_gap"],
            "fed_rate": _state_store["fed_rate"],
            "data_points": len(df),
            "is_synthetic": use_synthetic,
        }

    except Exception as e:
        logger.exception("Failed to refresh state")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/current")
async def get_current_state():
    """
    Return the current inferred macro state and structure metrics.
    """
    if not _state_store:
        raise HTTPException(
            status_code=404,
            detail="State not initialized. Call POST /api/state/refresh first.",
        )

    latest = _state_store["latest_state"]
    struct = _state_store["structure"]

    return {
        "latest_date": _state_store["latest_date"],
        "is_synthetic": _state_store.get("is_synthetic", False),
        "mu_T": latest["mu_T"],
        "P_T": latest["P_T"],
        "stress_score": latest["stress_score"],
        "regime_label": latest["regime_label"],
        "crisis_threshold": latest["crisis_threshold"],
        "inflation_gap": _state_store.get("inflation_gap", 0.0),
        "fed_rate": _state_store.get("fed_rate", 0.03),
        "metrics": struct["metrics"],
        "correlation_matrix": [row.tolist() for row in struct["R_t"]],
        "correlation_labels": struct["labels"],
        "eigenvalues": struct["eigenvalues"],
    }


# ── Historical snapshot builder ─────────────────────────────────


def _build_historical_snapshots(kalman, struct, dm) -> dict:
    """
    Build a dict of monthly state snapshots for the historical timeline.

    Each snapshot has the same schema as GET /api/state/current so that
    all frontend panels can consume any snapshot interchangeably.

    Keys are date strings ``"YYYY-MM-DD"`` (month-end).
    """
    import pandas as pd

    Z_history = struct["Z_history"]  # DataFrame with DatetimeIndex
    monthly_dates = Z_history.index
    smoothed_means = kalman.smoothed_means   # (T, n)
    smoothed_covs = kalman.smoothed_covs     # (T, n, n)
    T_kalman = len(smoothed_means)

    # Stress series statistics (for scoring)
    stress_series = smoothed_means[:, 0]
    stress_mean = float(np.mean(stress_series))
    stress_std = float(np.std(stress_series)) + 1e-10
    crisis_threshold = float(np.percentile(stress_series, 95))

    # Monthly macro series
    inflation_gap_series = dm.get_inflation_gap()
    fed_rate_series = dm.get_fed_rate_series()

    # Correlation snapshots from structure features
    corr_snapshots = struct.get("monthly_corr_snapshots", {})
    labels = struct.get("labels", [])

    snapshots: dict = {}

    for t, date in enumerate(monthly_dates):
        if t >= T_kalman:
            break

        mu_t = smoothed_means[t]
        P_t = smoothed_covs[t]

        # Stress score (z-scored against full history)
        stress_score = float((mu_t[0] - stress_mean) / stress_std)

        # Regime label
        from app.config import settings as _s

        if stress_score < _s.regime_threshold_fragile:
            regime_label = "Normal"
        elif stress_score < _s.regime_threshold_crisis:
            regime_label = "Fragile"
        else:
            regime_label = "Crisis"

        # Inflation gap & fed rate at this date
        infl_gap = _lookup_nearest(inflation_gap_series, date)
        fr = _lookup_nearest(fed_rate_series, date)

        # Correlation data for this month
        month_key = date.strftime("%Y-%m")
        corr_data = corr_snapshots.get(month_key, {})

        # Metrics from Z_history row
        metrics: dict = {}
        for col in Z_history.columns:
            val = Z_history.loc[date, col]
            metrics[col] = float(val) if not (np.isnan(val) if np.isscalar(val) else False) else 0.0

        date_str = date.strftime("%Y-%m-%d")
        snapshots[date_str] = {
            "date": date_str,
            "latest_date": date_str,
            "is_synthetic": False,
            "mu_T": mu_t.tolist(),
            "P_T": P_t.tolist(),
            "stress_score": stress_score,
            "regime_label": regime_label,
            "crisis_threshold": crisis_threshold,
            "inflation_gap": infl_gap,
            "fed_rate": fr,
            "metrics": metrics,
            "correlation_matrix": corr_data.get("correlation_matrix", []),
            "correlation_labels": labels,
            "eigenvalues": corr_data.get("eigenvalues", []),
        }

    return snapshots


def _lookup_nearest(series, date) -> float:
    """Look up the nearest value in a pandas Series by date (ffill)."""
    if series is None or len(series) == 0:
        return 0.0
    try:
        idx = series.index.get_indexer([date], method="ffill")[0]
        if 0 <= idx < len(series):
            val = float(series.iloc[idx])
            return val if not np.isnan(val) else 0.0
    except Exception:
        pass
    return 0.0
