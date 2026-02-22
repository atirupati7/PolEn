"""
API routes for historical data browsing.

Provides endpoints to explore pre-computed monthly snapshots of the full
macro state — Kalman-filtered latent factors, cross-asset structure,
and economic indicators — at any point in the available history.

This enables the "Historical Mode" in the frontend where users can scrub
through time and see all indicators update in sync.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from app.api.routes_state import get_state_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/historical", tags=["historical"])


@router.get("/dates")
async def get_available_dates():
    """
    Return the sorted list of available monthly snapshot dates.

    Response:
        dates:  list of "YYYY-MM-DD" strings
        start:  earliest date
        end:    latest date
        count:  number of snapshots
    """
    store = get_state_store()
    if not store or "historical_snapshots" not in store:
        raise HTTPException(
            status_code=404,
            detail="Historical data not loaded. Call POST /api/state/refresh first.",
        )

    snapshots = store["historical_snapshots"]
    dates = sorted(snapshots.keys())

    return {
        "dates": dates,
        "start": dates[0] if dates else None,
        "end": dates[-1] if dates else None,
        "count": len(dates),
    }


@router.get("/state")
async def get_historical_state(
    date: str = Query(
        ...,
        description="Target date (YYYY-MM-DD). Returns closest available monthly snapshot.",
    ),
):
    """
    Return the full macro state snapshot at (or nearest to) the given date.

    The response schema matches GET /api/state/current so that all frontend
    panels (cross-asset structure, Fed policy monitor, latent state vector,
    eigenvalue spectrum, etc.) can consume it interchangeably.
    """
    store = get_state_store()
    if not store or "historical_snapshots" not in store:
        raise HTTPException(status_code=404, detail="Historical data not loaded.")

    snapshots = store["historical_snapshots"]

    # Exact match
    if date in snapshots:
        return snapshots[date]

    # Find closest date
    dates = sorted(snapshots.keys())
    if not dates:
        raise HTTPException(status_code=404, detail="No historical snapshots available.")

    try:
        target = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {date}. Expected YYYY-MM-DD.",
        )

    closest = min(
        dates, key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d") - target)
    )

    result = {**snapshots[closest]}
    result["requested_date"] = date
    result["actual_date"] = closest
    return result
