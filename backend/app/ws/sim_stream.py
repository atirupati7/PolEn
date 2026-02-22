"""
WebSocket handler for live Monte Carlo simulation streaming.

Client sends simulation parameters, server streams per-step results
with configurable delay for animation.
"""

import asyncio
import json
import logging
from typing import Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from app.api.routes_state import get_state_store
from app.config import settings
from app.models.monte_carlo import MonteCarloEngine

logger = logging.getLogger(__name__)


async def handle_simulation(websocket: WebSocket):
    """
    WebSocket handler for live Monte Carlo simulation.

    Client sends JSON:
    {
        "delta_bps": -50|0|50|custom,
        "alpha": ..., "beta": ..., "gamma": ..., "lambda": ...,
        "N": 5000,
        "H": 24,
        "speed_ms": 120,
        "shocks": {"credit": k, "vol": k, "rate": k},
        "regime_switching": true
    }

    Server streams JSON per step:
    {
        "step": k,
        "H": H,
        "stress_fan": {"p5": .., "p25": .., "p50": .., "p75": .., "p95": ..},
        "growth_fan": {...},
        "crisis_prob": ..,
        "es95_stress": ..,
        "spaghetti": [{"id": 0, "stress": ..}, ...]
    }
    """
    await websocket.accept()

    try:
        while True:
            # Wait for simulation request
            data = await websocket.receive_text()
            params = json.loads(data)

            logger.info(f"WS simulation request: {params}")

            store = get_state_store()
            if not store:
                await websocket.send_json({
                    "error": "State not initialized. Refresh first.",
                    "done": True,
                })
                continue

            # Parse parameters
            delta_bps = float(params.get("delta_bps", 0))
            N = int(params.get("N", settings.mc_default_paths))
            H = int(params.get("H", settings.mc_default_horizon))
            speed_ms = int(params.get("speed_ms", settings.mc_default_speed_ms))
            shocks = params.get("shocks", None)
            regime_switching = params.get("regime_switching", True)
            start_date = params.get("start_date", None)

            # Clamp parameters
            N = max(500, min(10000, N))
            H = max(6, min(36, H))
            speed_ms = max(20, min(2000, speed_ms))

            kalman = store["kalman"]
            latest = store["latest_state"]

            # Optional: override init state from historical snapshot
            init_mu = np.array(latest["mu_T"], dtype=np.float64)
            init_P = np.array(latest["P_T"], dtype=np.float64)
            if start_date:
                snapshots = store.get("historical_snapshots", {})
                snap = _resolve_snapshot(snapshots, start_date)
                if snap is not None:
                    init_mu = np.array(snap.get("mu_T", init_mu), dtype=np.float64)
                    init_P = np.array(snap.get("P_T", init_P), dtype=np.float64)
                    logger.info(f"WS simulation init from historical snapshot: requested={start_date} actual={snap.get('date')}")

            try:
                engine = MonteCarloEngine(
                    A_daily=kalman.A,
                    B_daily=kalman.B,
                    Q_daily=kalman.Q,
                    mu_T=init_mu,
                    P_T=init_P,
                    crisis_threshold=latest["crisis_threshold"],
                    stress_std=latest["stress_std"],
                )

                # Run full simulation (pre-compute all steps)
                steps = engine.simulate_streaming(
                    delta_bps=delta_bps,
                    N=N,
                    H=H,
                    shocks=shocks,
                    regime_switching=regime_switching,
                    seed=np.random.randint(0, 100000),
                )

                # Stream results step by step with delay
                for idx, step_data in enumerate(steps):
                    step_data["done"] = False
                    # First step includes initial state so frontend can show deviations
                    if idx == 0:
                        step_data["initial_mu"] = init_mu.tolist()
                        step_data["start_date"] = start_date or store.get("latest_date", "")
                    await websocket.send_json(step_data)
                    await asyncio.sleep(speed_ms / 1000.0)

                # Send completion signal
                await websocket.send_json({"done": True, "H": H})

            except Exception as e:
                logger.exception("Simulation error")
                await websocket.send_json({
                    "error": str(e),
                    "done": True,
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


def _resolve_snapshot(snapshots: dict, date_str: str) -> Optional[dict]:
    """Return the snapshot closest to the requested date string (YYYY-MM-DD)."""
    if not snapshots:
        return None

    if date_str in snapshots:
        return snapshots[date_str]

    try:
        target = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

    dates = sorted(snapshots.keys())
    if not dates:
        return None

    closest = min(
        dates, key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d") - target)
    )
    return snapshots.get(closest)
