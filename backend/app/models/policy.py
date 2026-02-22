"""
Policy recommendation engine.

Evaluates Ease/Hold/Tighten by minimizing a stochastic loss function
using Monte Carlo expectations + tail risk penalty (Expected Shortfall).
"""

import logging
from typing import Optional

import numpy as np

from app.models.monte_carlo import MonteCarloEngine

logger = logging.getLogger(__name__)

# Standard policy actions (delta bps)
POLICY_ACTIONS = {
    "Ease": -50,
    "Hold": 0,
    "Tighten": 50,
}


def recommend_policy(
    engine: MonteCarloEngine,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    lam: float = 1.0,
    N: int = 5000,
    H: int = 24,
    shocks: dict = None,
    regime_switching: bool = True,
    delta_bps_custom: Optional[float] = None,
) -> dict:
    """
    Evaluate all policy actions and recommend the one with minimum loss.

    Loss = alpha * mean_stress + beta * growth_penalty + gamma * ES95 + lambda * crisis_end

    Args:
        engine: Initialized MonteCarloEngine
        alpha: Weight on average stress
        beta: Weight on growth penalty (negative growth)
        gamma: Weight on ES95 (tail risk)
        lam: Weight on terminal crisis probability
        N: Number of Monte Carlo paths
        H: Horizon in months
        shocks: Shock injection dict
        regime_switching: Whether to use regime switching
        delta_bps_custom: Optional custom bps to evaluate as 4th option

    Returns:
        dict with comparison table and recommendation
    """
    actions = dict(POLICY_ACTIONS)
    if delta_bps_custom is not None:
        actions[f"Custom({delta_bps_custom:+.0f})"] = delta_bps_custom

    results = {}
    for name, bps in actions.items():
        logger.info(f"Evaluating policy: {name} ({bps:+d} bps)")
        result = engine.evaluate_policy(
            delta_bps=bps,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lam=lam,
            N=N,
            H=H,
            shocks=shocks,
            regime_switching=regime_switching,
            seed=42 + hash(name) % 1000,
        )
        result["action"] = name
        results[name] = result

    # Find recommendation
    min_loss = float("inf")
    recommended = None
    for name, r in results.items():
        if r["total_loss"] < min_loss:
            min_loss = r["total_loss"]
            recommended = name

    # Generate explanation
    rec = results[recommended]
    explanation = _generate_explanation(recommended, rec, results)

    return {
        "recommended_action": recommended,
        "recommended_bps": actions[recommended],
        "explanation": explanation,
        "comparison": [
            {
                "action": r["action"],
                "delta_bps": r["delta_bps"],
                "mean_stress": round(r["mean_stress"], 4),
                "mean_growth_penalty": round(r["mean_growth_penalty"], 4),
                "mean_es95": round(r["mean_es95"], 4),
                "crisis_end": round(r["crisis_end"], 4),
                "total_loss": round(r["total_loss"], 4),
            }
            for r in results.values()
        ],
        "weights": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "lambda": lam,
        },
    }


def _generate_explanation(
    recommended: str,
    rec_result: dict,
    all_results: dict,
) -> str:
    """Generate a 1-2 sentence explanation for the recommendation."""
    # Find which component most differentiates the recommendation
    components = ["mean_stress", "mean_growth_penalty", "mean_es95", "crisis_end"]
    component_names = {
        "mean_stress": "average stress",
        "mean_growth_penalty": "growth downside risk",
        "mean_es95": "tail risk (ES95)",
        "crisis_end": "terminal crisis probability",
    }

    # Check which component the recommended action minimizes most vs alternatives
    best_margin = 0
    best_component = "mean_stress"
    for comp in components:
        rec_val = rec_result[comp]
        others = [r[comp] for name, r in all_results.items() if name != recommended]
        if others:
            margin = min(others) - rec_val
            if margin > best_margin:
                best_margin = margin
                best_component = comp

    explanation = (
        f"'{recommended}' is recommended because it minimizes the overall stochastic loss "
        f"(L={rec_result['total_loss']:.4f}), with particularly strong performance on "
        f"{component_names[best_component]} ({rec_result[best_component]:.4f}). "
        f"Terminal crisis probability: {rec_result['crisis_end']:.1%}."
    )

    return explanation
