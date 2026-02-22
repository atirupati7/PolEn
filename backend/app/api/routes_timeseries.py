"""
API routes for historical time-series data.

Serves monthly-resampled cross-asset data and Kalman-estimated latent
factors so the frontend can render the full historical picture with
simulation overlays.
"""

import logging
import math

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.api.routes_state import get_state_store
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/historical", tags=["timeseries"])

# ── Series we expose to the frontend ────────────────────────────

_SERIES_META: dict[str, str] = {
    "r_spx":          "SPX Returns",
    "DGS2":           "2Y Yield (%)",
    "DGS10":          "10Y Yield (%)",
    "vix_level":      "VIX",
    "cs":             "Credit Spread",
    "slope":          "Yield Curve Slope",
    "inflation_yoy":  "Inflation YoY",
    "fed_rate":       "Fed Funds Rate",
}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@router.get("/timeseries")
async def get_timeseries(years: int = Query(default=10, ge=1, le=40)):
    """
    Return historical monthly time-series data for charting.

    Query params:
        years: how many years of history to return (default 10, max 40).

    Response schema::

        {
            "dates": ["YYYY-MM-DD", ...],
            "series": {
                "r_spx": {"label": "SPX Returns", "values": [...]},
                ...
            },
            "latent_factors": {
                "stress":    [...],
                "liquidity": [...],
                "growth":    [...]
            },
            "regime_labels":     ["Normal", ...],
            "crisis_probability": [0.05, ...],
            "stress_mean": ...,
            "stress_std":  ...
        }
    """
    store = get_state_store()
    if not store:
        raise HTTPException(
            status_code=404,
            detail="State not initialized. Call POST /api/state/refresh first.",
        )

    df = store["df"]
    kalman = store["kalman"]
    struct = store["structure"]

    Z_history = struct["Z_history"]  # DataFrame with DatetimeIndex (monthly)
    smoothed = kalman.smoothed_means  # (T, n)
    T = min(len(Z_history), len(smoothed))

    # Slice to at most `years` worth of months
    max_months = years * 12
    start = max(0, T - max_months)

    dates_idx = Z_history.index[start:T]
    dates = [d.strftime("%Y-%m-%d") for d in dates_idx]

    # ── Raw series (monthly last values) ────────────────────────
    monthly_df = df.resample("M").last()

    raw_series: dict = {}
    for col, label in _SERIES_META.items():
        if col not in monthly_df.columns:
            continue
        aligned = monthly_df[col].reindex(dates_idx, method="ffill")
        raw_series[col] = {
            "label": label,
            "values": [
                round(float(v), 6) if (v is not None and not np.isnan(v)) else None
                for v in aligned.values
            ],
        }

    # ── Latent factors ──────────────────────────────────────────
    sm = smoothed[start:T]
    n_dim = sm.shape[1]

    latent: dict[str, list[float]] = {
        "stress": [round(float(sm[t, 0]), 6) for t in range(len(sm))],
    }
    if n_dim > 1:
        latent["liquidity"] = [round(float(sm[t, 1]), 6) for t in range(len(sm))]
    if n_dim > 2:
        latent["growth"] = [round(float(sm[t, 2]), 6) for t in range(len(sm))]

    # ── Regime labels + crisis probability (sigmoid of stress) ──
    stress_full = smoothed[:T, 0]
    stress_mean = float(np.mean(stress_full))
    stress_std = float(np.std(stress_full)) + 1e-10

    regime_labels: list[str] = []
    crisis_prob: list[float] = []

    for t in range(start, T):
        z = (smoothed[t, 0] - stress_mean) / stress_std
        # Regime
        if z < settings.regime_threshold_fragile:
            regime_labels.append("Normal")
        elif z < settings.regime_threshold_crisis:
            regime_labels.append("Fragile")
        else:
            regime_labels.append("Crisis")
        # Crisis probability via sigmoid centred on fragile/crisis boundary
        cp = _sigmoid(2.0 * (z - settings.regime_threshold_crisis))
        crisis_prob.append(round(cp, 4))

    return {
        "dates": dates,
        "series": raw_series,
        "latent_factors": latent,
        "regime_labels": regime_labels,
        "crisis_probability": crisis_prob,
        "stress_mean": round(stress_mean, 6),
        "stress_std": round(stress_std, 6),
        "regime_threshold_crisis": float(settings.regime_threshold_crisis),
    }
