"""
Evaluation script — compare RL agent vs heuristic optimizer.

Runs N episodes in ``MacroPolicyEnvV2`` and collects:
- Average episode reward
- Crisis frequency (% of steps where stress > threshold)
- Tail risk metrics (ES95 of stress)
- Inflation gap statistics
- Action distribution statistics

Run standalone:
    python -m app.rl.evaluate
"""

import logging
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


def _build_env_and_models():
    """Shared helper to construct V2 env, RL engine, and heuristic engine."""
    from app.data.manager import DataManager
    from app.models.structure_features import compute_structure_features
    from app.models.kalman_em import KalmanEM
    from app.models.regime import RegimeModel
    from app.models.monte_carlo import MonteCarloEngine
    from app.policy.rl_policy import RLPolicyEngine
    from app.environment.macro_env import MacroPolicyEnvV2
    from app.environment.reward import RewardModule

    # ── Data ──
    dm = DataManager()
    df = dm.load_local_data()
    if df is None or len(df) == 0:
        df = dm.download_data(synthetic=True)

    struct = compute_structure_features(df, window=settings.rolling_cov_window)
    Z_history = struct["Z_history"].values

    kalman = KalmanEM(latent_dim=settings.kalman_latent_dim)
    kalman.fit(Z_history)

    smoothed_3d = kalman.smoothed_means
    latest = kalman.get_latest_state()

    # 4-dim state
    smoothed_4d = dm.build_state_matrix(smoothed_3d)

    # Eigen history
    eigen_history = np.array(
        struct["Z_history"][["lambda1"]].values, dtype=np.float32
    )
    if eigen_history.ndim == 1:
        eigen_history = eigen_history[:, np.newaxis]
    if eigen_history.shape[1] < 3:
        pad = np.zeros(
            (eigen_history.shape[0], 3 - eigen_history.shape[1]), dtype=np.float32
        )
        eigen_history = np.hstack([eigen_history, pad])

    # Fed rate history
    fed_rate = dm.get_fed_rate_series()
    fed_rate_vals = fed_rate.values[-len(smoothed_3d):] if len(fed_rate) >= len(smoothed_3d) else np.full(len(smoothed_3d), 0.03)

    min_len = min(len(smoothed_4d), len(eigen_history), len(fed_rate_vals))
    hist_states = smoothed_4d[-min_len:]
    hist_eigen = eigen_history[-min_len:]
    hist_fed = fed_rate_vals[-min_len:]

    # Transition model
    if settings.transition_model_type == "neural":
        from app.models.neural_transition import NeuralTransitionModel
        transition = NeuralTransitionModel()
        transition.load()
    else:
        from app.models.linear_transition import LinearStateTransition
        transition = LinearStateTransition(A=kalman.A, B=kalman.B)

    regime_model = RegimeModel()
    reward_mod = RewardModule()

    env = MacroPolicyEnvV2(
        transition_model=transition,
        regime_model=regime_model,
        historical_states=hist_states,
        eigenvalues_history=hist_eigen,
        fed_rate_history=hist_fed,
        crisis_threshold=latest["crisis_threshold"],
        reward_module=reward_mod,
        seed=settings.ppo_seed,
    )

    # RL engine
    rl_engine = RLPolicyEngine()
    rl_engine.load_model()

    # MC engine for heuristic
    mc_engine = MonteCarloEngine(
        A_daily=kalman.A,
        B_daily=kalman.B,
        Q_daily=kalman.Q,
        mu_T=np.array(latest["mu_T"]),
        P_T=np.array(latest["P_T"]),
        crisis_threshold=latest["crisis_threshold"],
        stress_std=latest["stress_std"],
    )

    return env, rl_engine, mc_engine, kalman, latest


def _run_episodes(env, policy_fn, n_episodes: int = 50, seed: int = 0):
    """
    Run *n_episodes* and collect per-step metrics.

    Returns:
        dict with aggregated statistics.
    """
    episode_rewards: list[float] = []
    all_stresses: list[float] = []
    all_actions: list[float] = []
    all_inflation_gaps: list[float] = []
    crisis_steps = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            all_stresses.append(info["stress"])
            all_actions.append(info.get("delta_rate", info.get("delta_bps", 0) / 100))
            all_inflation_gaps.append(info.get("inflation_gap", 0.0))
            if info.get("crisis", False):
                crisis_steps += 1
            total_steps += 1
            done = terminated or truncated
        episode_rewards.append(ep_reward)

    stresses = np.array(all_stresses)
    actions = np.array(all_actions)
    infl_gaps = np.array(all_inflation_gaps)

    # ES95 of stress
    if len(stresses) > 0:
        p95 = np.percentile(stresses, 95)
        tail = stresses[stresses >= p95]
        es95 = float(np.mean(tail)) if len(tail) > 0 else float(p95)
    else:
        es95 = 0.0

    return {
        "avg_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "crisis_frequency": crisis_steps / max(total_steps, 1),
        "es95_stress": es95,
        "mean_stress": float(np.mean(stresses)) if len(stresses) else 0.0,
        "mean_inflation_gap": float(np.mean(infl_gaps)) if len(infl_gaps) else 0.0,
        "mean_abs_action_pct": float(np.mean(np.abs(actions))) if len(actions) else 0.0,
        "std_action_pct": float(np.std(actions)) if len(actions) else 0.0,
        "n_episodes": n_episodes,
        "total_steps": total_steps,
    }


def evaluate(n_episodes: int = 50, seed: int = 0) -> dict:
    """
    Full evaluation: run both RL and heuristic agents, return comparison.

    Returns:
        dict with keys "rl", "heuristic", "comparison_summary".
    """
    logger.info(f"Starting evaluation ({n_episodes} episodes each) …")

    env, rl_engine, mc_engine, kalman, latest = _build_env_and_models()

    # ── RL agent episodes ──
    logger.info("Evaluating RL agent …")

    def rl_policy(obs: np.ndarray) -> np.ndarray:
        result = rl_engine.predict(obs, deterministic=True)
        return np.array([result["action_raw"]], dtype=np.float32)

    rl_stats = _run_episodes(env, rl_policy, n_episodes=n_episodes, seed=seed)

    # ── Heuristic agent episodes (maps stress → discrete action) ──
    logger.info("Evaluating heuristic agent …")

    def heuristic_policy(obs: np.ndarray) -> np.ndarray:
        """Simple heuristic: choose Ease/Hold/Tighten based on stress."""
        stress = obs[0]
        if stress > 1.0:
            delta_pct = -0.25  # ease 25 bps
        elif stress < -0.5:
            delta_pct = 0.25   # tighten 25 bps
        else:
            delta_pct = 0.0    # hold
        # Map % points to [−1, +1]
        raw = delta_pct / settings.gym_max_rate_step if settings.gym_max_rate_step > 0 else 0.0
        return np.array([np.clip(raw, -1.0, 1.0)], dtype=np.float32)

    heuristic_stats = _run_episodes(
        env, heuristic_policy, n_episodes=n_episodes, seed=seed
    )

    # ── Summary ──
    reward_diff = rl_stats["avg_episode_reward"] - heuristic_stats["avg_episode_reward"]
    crisis_diff = rl_stats["crisis_frequency"] - heuristic_stats["crisis_frequency"]

    comparison = {
        "reward_advantage": reward_diff,
        "crisis_frequency_delta": crisis_diff,
        "rl_better_reward": reward_diff > 0,
        "rl_fewer_crises": crisis_diff < 0,
    }

    result = {
        "rl": rl_stats,
        "heuristic": heuristic_stats,
        "comparison_summary": comparison,
    }

    logger.info(
        f"Evaluation complete — RL avg reward: {rl_stats['avg_episode_reward']:.2f}, "
        f"Heuristic avg reward: {heuristic_stats['avg_episode_reward']:.2f}, "
        f"Δ = {reward_diff:+.2f}"
    )

    return result


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = evaluate()
    print(json.dumps(result, indent=2))
