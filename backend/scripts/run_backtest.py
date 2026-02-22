#!/usr/bin/env python
"""
CLI entry-point for the PolEn offline backtest.

Usage (from backend/):
    python -m scripts.run_backtest
    python scripts/run_backtest.py --window 120 --horizon 12 --paths 2000
    python scripts/run_backtest.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure backend/ is on sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PolEn offline backtest — rolling-window out-of-sample evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.run_backtest                     # defaults
  python -m scripts.run_backtest --window 60         # shorter estimation window
  python -m scripts.run_backtest --paths 5000        # more MC paths (slower)
  python -m scripts.run_backtest --step 3            # advance 3 months per step
  python -m scripts.run_backtest --output results    # save to results/
        """,
    )

    parser.add_argument(
        "--window", "-w",
        type=int, default=120,
        help="Rolling estimation window in months (default: 120)",
    )
    parser.add_argument(
        "--horizon", "-H",
        type=int, default=12,
        help="Forecast horizon in months (default: 12)",
    )
    parser.add_argument(
        "--paths", "-N",
        type=int, default=2000,
        help="Number of Monte Carlo paths per step (default: 2000)",
    )
    parser.add_argument(
        "--step", "-s",
        type=int, default=1,
        help="Advance window by this many months per step (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int, default=12345,
        help="Base random seed for reproducibility (default: 12345)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # ── Logging ──
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quiet down noisy libraries
    for lib in ("numba", "urllib3", "httpcore", "matplotlib"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logger = logging.getLogger("backtest.cli")

    # ── Banner ──
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║         PolEn Offline Backtest Runner            ║")
    logger.info("╠══════════════════════════════════════════════════╣")
    logger.info(f"║  Window  : {args.window:>5} months                       ║")
    logger.info(f"║  Horizon : {args.horizon:>5} months                       ║")
    logger.info(f"║  Paths   : {args.paths:>5}                              ║")
    logger.info(f"║  Step    : {args.step:>5} month(s)                      ║")
    logger.info(f"║  Seed    : {args.seed:>5}                              ║")
    logger.info(f"║  Output  : {args.output:<30}       ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    # ── Run ──
    from app.backtest.backtest import BacktestEngine

    t0 = time.time()

    engine = BacktestEngine(
        rolling_window=args.window,
        forecast_horizon=args.horizon,
        mc_paths=args.paths,
        results_dir=args.output,
        seed=args.seed,
    )

    results = engine.run(step_every=args.step)

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Forecasts generated : {len(results.forecast_records)}")
    logger.info(f"  Time elapsed        : {minutes}m {seconds:.1f}s")
    logger.info("")

    if len(results.forecast_records) > 0:
        logger.info("┌─── Crisis Probability ───────────────────────────")
        logger.info(f"│  Brier Score : {results.brier_score:.4f}")
        logger.info(f"│  AUC         : {results.auc_crisis:.4f}")
        logger.info(f"│  Log Score   : {results.log_score:.4f}")
        logger.info("├─── ES95 Validation ──────────────────────────────")
        logger.info(f"│  Hit Ratio        : {results.es95_hit_ratio:.4f} (target: 0.05)")
        logger.info(f"│  Kupiec p-value   : {results.kupiec_pvalue:.4f}")
        logger.info(f"│  Christoff. p-val : {results.christoffersen_pvalue:.4f}")
        logger.info("├─── Regime Classification ────────────────────────")
        logger.info(f"│  Precision : {results.regime_precision:.3f}")
        logger.info(f"│  Recall    : {results.regime_recall:.3f}")
        logger.info(f"│  F1        : {results.regime_f1:.3f}")
        logger.info(f"│  AUC       : {results.regime_auc:.3f}")
        logger.info("├─── Policy Comparison ────────────────────────────")
        logger.info(f"│  Avg Loss (active)  : {results.policy_avg_stress_active:.4f}")
        logger.info(f"│  Avg Loss (passive) : {results.policy_avg_stress_passive:.4f}")
        logger.info(f"│  ES95 (active)      : {results.policy_es95_active:.4f}")
        logger.info(f"│  ES95 (passive)     : {results.policy_es95_passive:.4f}")
        logger.info("└─────────────────────────────────────────────────")

        # Report episodes
        for ep_name, ep_data in results.episode_results.items():
            if "error" not in ep_data:
                logger.info(f"  Episode {ep_name}: max_crisis_prob={ep_data.get('max_crisis_prob', 'N/A')}")

    logger.info(f"\nResults saved to: {Path(args.output).resolve()}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
