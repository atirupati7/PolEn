"""
Reward module for the inflation-aware Fed-policy RL environment.

Implements the dual-mandate reward:

    R_t = -(w₁·stress² + w₂·inflation_gap² + w₃·crisis_prob
           + w₄·rate_change² + w₅·taylor_deviation²)

All weights and the inflation target are read from ``settings``.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Breakdown of a single-step reward for logging / debugging."""
    stress_penalty: float = 0.0
    inflation_penalty: float = 0.0
    crisis_penalty: float = 0.0
    rate_change_penalty: float = 0.0
    taylor_penalty: float = 0.0
    total: float = 0.0


class RewardModule:
    """
    Configurable reward function for the macro-policy environment.

    Parameters
    ----------
    w_stress        : weight on stress²
    w_inflation     : weight on (inflation − target)²
    w_crisis        : weight on crisis indicator
    w_rate_change   : weight on (Δrate)²  (policy inertia penalty)
    w_taylor        : weight on (rate − taylor_rate)²
    inflation_target: target annual CPI inflation (e.g. 0.02)
    """

    def __init__(
        self,
        w_stress: Optional[float] = None,
        w_inflation: Optional[float] = None,
        w_crisis: Optional[float] = None,
        w_rate_change: Optional[float] = None,
        w_taylor: Optional[float] = None,
        inflation_target: Optional[float] = None,
    ) -> None:
        self.w_stress = w_stress if w_stress is not None else settings.reward_w_stress
        self.w_inflation = w_inflation if w_inflation is not None else settings.reward_w_inflation
        self.w_crisis = w_crisis if w_crisis is not None else settings.reward_w_crisis
        self.w_rate_change = w_rate_change if w_rate_change is not None else settings.reward_w_rate_change
        self.w_taylor = w_taylor if w_taylor is not None else settings.reward_w_taylor
        self.inflation_target = inflation_target if inflation_target is not None else settings.reward_inflation_target

    def compute(
        self,
        stress: float,
        inflation: float,
        crisis_prob: float,
        rate_change: float,
        current_rate: float = 0.0,
        growth: float = 0.0,
    ) -> RewardComponents:
        """
        Compute the reward and its components.

        Args:
            stress:       current stress level (state[0])
            inflation:    current inflation_gap (state[3] or inflation_yoy)
            crisis_prob:  probability of being in crisis (0 or 1, or soft)
            rate_change:  Δ federal funds rate this step (% points)
            current_rate: current federal funds rate level (for Taylor rule)
            growth:       current growth indicator (for Taylor rule)

        Returns:
            RewardComponents with individual terms and total.
        """
        inflation_gap = inflation - self.inflation_target if abs(inflation) > 0.5 else inflation
        # If inflation is already a gap (small magnitude), use directly
        # If it's raw YoY (like 0.03), subtract target

        stress_pen = self.w_stress * stress ** 2
        inflation_pen = self.w_inflation * inflation_gap ** 2
        crisis_pen = self.w_crisis * crisis_prob
        rate_pen = self.w_rate_change * rate_change ** 2

        # Taylor rule deviation (optional)
        taylor_pen = 0.0
        if self.w_taylor > 0:
            # Simplified Taylor rule: r* = r_neutral + 0.5*(inflation - target) + 0.5*growth
            r_neutral = 0.02  # 2% real neutral rate
            taylor_rate = r_neutral + 0.5 * inflation_gap + 0.5 * growth
            taylor_pen = self.w_taylor * (current_rate - taylor_rate) ** 2

        total = -(stress_pen + inflation_pen + crisis_pen + rate_pen + taylor_pen)

        return RewardComponents(
            stress_penalty=stress_pen,
            inflation_penalty=inflation_pen,
            crisis_penalty=crisis_pen,
            rate_change_penalty=rate_pen,
            taylor_penalty=taylor_pen,
            total=total,
        )

    def __repr__(self) -> str:
        return (
            f"RewardModule(w_stress={self.w_stress}, w_inflation={self.w_inflation}, "
            f"w_crisis={self.w_crisis}, w_rate_change={self.w_rate_change}, "
            f"w_taylor={self.w_taylor}, target={self.inflation_target})"
        )
