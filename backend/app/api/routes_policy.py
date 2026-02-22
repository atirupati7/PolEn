"""
API routes for policy recommendation.

Supports two modes (toggled at runtime):
  - "heuristic"  → existing Monte Carlo optimizer
  - "rl"         → PPO-trained reinforcement learning agent
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.routes_state import get_state_store
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["policy"])

# ── Runtime policy mode (mutable server state) ────────────────────
_policy_mode: str = settings.policy_mode  # "heuristic" | "rl"


def get_policy_mode() -> str:
    return _policy_mode


# ── Request / response models ─────────────────────────────────────

class RecommendRequest(BaseModel):
    alpha: float = Field(default=1.0, ge=0, le=10, description="Stress weight")
    beta: float = Field(default=1.0, ge=0, le=10, description="Growth penalty weight")
    gamma: float = Field(default=1.0, ge=0, le=10, description="Tail risk (ES95) weight")
    lam: float = Field(default=1.0, ge=0, le=10, description="Crisis end weight", alias="lambda")
    N: int = Field(default=5000, ge=500, le=10000, description="Monte Carlo paths")
    H: int = Field(default=24, ge=6, le=36, description="Horizon months")
    delta_bps_custom: Optional[float] = Field(default=None, description="Custom bps to evaluate")
    shocks: Optional[dict] = Field(default=None, description="Shock injections")
    regime_switching: bool = Field(default=True, description="Enable regime switching")

    class Config:
        populate_by_name = True


class SetModeRequest(BaseModel):
    mode: str = Field(
        ...,
        pattern="^(heuristic|rl)$",
        description="Policy engine mode: 'heuristic' or 'rl'",
    )


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/policy/set_mode")
async def set_policy_mode(req: SetModeRequest):
    """Switch between heuristic and RL policy engines at runtime."""
    global _policy_mode
    old = _policy_mode
    _policy_mode = req.mode
    logger.info(f"Policy mode changed: {old} → {_policy_mode}")
    return {"mode": _policy_mode, "previous": old}


@router.get("/policy/mode")
async def get_mode():
    """Return the current policy engine mode."""
    return {"mode": _policy_mode}


@router.post("/policy/recommend")
async def recommend(req: RecommendRequest):
    """
    Run policy recommendation.

    Dispatches to either the heuristic Monte Carlo optimizer or the
    RL PPO agent depending on the current server mode.
    """
    store = get_state_store()
    if not store:
        raise HTTPException(
            status_code=404,
            detail="State not initialized. Call POST /api/state/refresh first.",
        )

    try:
        import numpy as np

        kalman = store["kalman"]
        latest = store["latest_state"]

        # ── RL mode ───────────────────────────────────────────────
        if _policy_mode == "rl":
            return await _recommend_rl(store, latest)

        # ── Heuristic mode (existing) ─────────────────────────────
        return await _recommend_heuristic(req, store, kalman, latest)

    except FileNotFoundError as e:
        logger.warning(f"RL model not found, falling back to heuristic: {e}")
        return await _recommend_heuristic(req, store, store["kalman"], store["latest_state"])
    except Exception as e:
        logger.exception("Policy recommendation failed")
        raise HTTPException(status_code=500, detail=str(e))


async def _recommend_heuristic(req, store, kalman, latest):
    """Original heuristic Monte Carlo recommendation."""
    import numpy as np
    from app.models.monte_carlo import MonteCarloEngine
    from app.models.policy import recommend_policy

    engine = MonteCarloEngine(
        A_daily=kalman.A,
        B_daily=kalman.B,
        Q_daily=kalman.Q,
        mu_T=np.array(latest["mu_T"]),
        P_T=np.array(latest["P_T"]),
        crisis_threshold=latest["crisis_threshold"],
        stress_std=latest["stress_std"],
    )

    result = recommend_policy(
        engine=engine,
        alpha=req.alpha,
        beta=req.beta,
        gamma=req.gamma,
        lam=req.lam,
        N=req.N,
        H=req.H,
        shocks=req.shocks,
        regime_switching=req.regime_switching,
        delta_bps_custom=req.delta_bps_custom,
    )
    result["engine"] = "heuristic"
    return result


async def _recommend_rl(store, latest):
    """RL PPO-based recommendation."""
    import numpy as np
    from app.policy.rl_policy import RLPolicyEngine

    rl = RLPolicyEngine()
    rl.load_model()

    # Build 4-dim state: [stress, liquidity, growth, inflation_gap]
    mu_T = np.array(latest["mu_T"], dtype=np.float32)
    stress_score = latest["stress_score"]

    # Inflation gap from store
    inflation_gap = store.get("inflation_gap", 0.0)
    fed_rate = store.get("fed_rate", 0.03)

    # Build 4-dim x_t
    x_t = np.append(mu_T, inflation_gap).astype(np.float32)

    # Determine regime index
    if stress_score < settings.regime_threshold_fragile:
        regime = 0
    elif stress_score < settings.regime_threshold_crisis:
        regime = 1
    else:
        regime = 2

    # Eigenvalues from structure features
    struct = store.get("structure", {})
    eigenvalues = np.array(struct.get("eigenvalues", [0, 0, 0])[:3], dtype=np.float32)

    result = rl.predict_from_state(
        x_t=x_t,
        regime=regime,
        last_action=0.0,
        fed_rate=fed_rate,
        eigenvalues=eigenvalues,
        deterministic=True,
    )
    result["engine"] = "rl"
    result["regime"] = ["Normal", "Fragile", "Crisis"][regime]
    result["stress_score"] = stress_score
    result["inflation_gap"] = inflation_gap
    result["fed_rate"] = fed_rate
    return result


@router.get("/policy/evaluation")
async def policy_evaluation():
    """
    Run a full RL-vs-heuristic evaluation (50 episodes each).

    This is compute-intensive; results are returned synchronously.
    """
    try:
        from app.rl.evaluate import evaluate
        result = evaluate(n_episodes=20, seed=42)
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model checkpoint not found: {e}",
        )
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))
