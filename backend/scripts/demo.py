"""
Demo script: refresh state, run recommendation, print results.

Usage:
    cd backend
    python -m scripts.demo
"""

import asyncio
import json
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    print("=" * 60)
    print("MacroState Control Room — Demo Script")
    print("=" * 60)
    print()

    # Step 1: Refresh state
    print("[1/3] Refreshing state (data + pipeline + Kalman)...")
    from app.api.routes_state import refresh_state
    result = await refresh_state(synthetic=True)
    print(f"  Latest date: {result['latest_date']}")
    print(f"  Regime: {result['regime_label']}")
    print(f"  Stress score: {result['stress_score']:.3f}")
    print(f"  Data points: {result['data_points']}")
    print(f"  Synthetic: {result['is_synthetic']}")
    print()

    # Step 2: Get current state
    print("[2/3] Current macro state:")
    from app.api.routes_state import get_current_state
    state = await get_current_state()
    print(f"  μ_T = {state['mu_T']}")
    print(f"  Regime: {state['regime_label']}")
    print(f"  Crisis threshold: {state['crisis_threshold']:.4f}")
    print(f"  Correlation labels: {state['correlation_labels']}")
    print(f"  Eigenvalues: {[f'{v:.3f}' for v in state['eigenvalues']]}")
    print(f"  Metrics:")
    for k, v in state['metrics'].items():
        print(f"    {k}: {v:.4f}")
    print()

    # Step 3: Run recommendation
    print("[3/3] Running policy recommendation (N=2000, H=24)...")
    from app.api.routes_policy import RecommendRequest, recommend
    req = RecommendRequest(
        alpha=1.0, beta=1.0, gamma=1.0, N=2000, H=24,
        regime_switching=True, **{"lambda": 1.0},
    )
    rec = await recommend(req)

    print(f"\n  Recommendation: {rec['recommended_action']} ({rec['recommended_bps']:+d} bps)")
    print(f"  Explanation: {rec['explanation']}")
    print()
    print("  Comparison Table:")
    print(f"  {'Action':<15} {'Δbps':>6} {'Stress':>10} {'Growth↓':>10} {'ES95':>10} {'Crisis%':>10} {'Loss':>10}")
    print("  " + "-" * 75)
    for row in rec['comparison']:
        mark = " ◀" if row['action'] == rec['recommended_action'] else ""
        print(
            f"  {row['action']:<15} {row['delta_bps']:>+6.0f} "
            f"{row['mean_stress']:>10.4f} {row['mean_growth_penalty']:>10.4f} "
            f"{row['mean_es95']:>10.4f} {row['crisis_end']:>10.4f} "
            f"{row['total_loss']:>10.4f}{mark}"
        )

    print()
    print("=" * 60)
    print("Demo complete. Start the server with:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
