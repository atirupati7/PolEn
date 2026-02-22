"""
PPO training script for the macro-policy RL agent.

Uses stable-baselines3 PPO with an MlpPolicy.  The environment is
``MacroPolicyEnvV2`` backed by a configurable transition model (linear
or neural) and inflation-aware dual-mandate reward.

Run standalone:
    python -m app.rl.train_ppo          # from backend/
    python -m app.rl.train_ppo --steps 200000 --seed 123

The trained policy is saved to ``models/checkpoints/ppo_policy.zip``.
TensorBoard logs go to ``runs/ppo_macro``.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


def _build_env():
    """Construct the MacroPolicyEnvV2 with the configured transition model."""
    from app.data.manager import DataManager
    from app.models.structure_features import compute_structure_features
    from app.models.kalman_em import KalmanEM
    from app.models.regime import RegimeModel
    from app.environment.macro_env import MacroPolicyEnvV2
    from app.environment.reward import RewardModule

    # ── Data ──
    dm = DataManager()
    df = dm.load_local_data()
    if df is None or len(df) == 0:
        df = dm.download_data(synthetic=True)

    # ── Structure features + Kalman ──
    struct = compute_structure_features(df, window=settings.rolling_cov_window)
    Z_history = struct["Z_history"].values

    kalman = KalmanEM(latent_dim=settings.kalman_latent_dim)
    kalman.fit(Z_history)

    smoothed_3d = kalman.smoothed_means  # (T, 3)

    # Build 4-dim state matrix
    smoothed_4d = dm.build_state_matrix(smoothed_3d)

    # Top-3 eigenvalues
    eigen_history = np.array(struct["Z_history"][["lambda1"]].values, dtype=np.float32)
    if eigen_history.ndim == 1:
        eigen_history = eigen_history[:, np.newaxis]
    if eigen_history.shape[1] < 3:
        pad = np.zeros((eigen_history.shape[0], 3 - eigen_history.shape[1]), dtype=np.float32)
        eigen_history = np.hstack([eigen_history, pad])

    # Fed rate history
    fed_rate = dm.get_fed_rate_series()
    fed_rate_vals = fed_rate.values[-len(smoothed_3d):] if len(fed_rate) >= len(smoothed_3d) else np.full(len(smoothed_3d), 0.03)

    # Align lengths
    min_len = min(len(smoothed_4d), len(eigen_history), len(fed_rate_vals))
    hist_states = smoothed_4d[-min_len:]
    hist_eigen = eigen_history[-min_len:]
    hist_fed = fed_rate_vals[-min_len:]

    # ── Transition model ──
    if settings.transition_model_type == "neural":
        from app.models.neural_transition import NeuralTransitionModel
        transition = NeuralTransitionModel()
        if not transition.is_available:
            raise FileNotFoundError(
                f"Neural transition checkpoint not found at {settings.neural_checkpoint_path}. "
                "Run  python -m app.models.neural_transition  first."
            )
        transition.load()
    else:
        from app.models.linear_transition import LinearStateTransition
        transition = LinearStateTransition(A=kalman.A, B=kalman.B)

    # ── Regime model ──
    regime = RegimeModel()

    # ── Crisis threshold ──
    latest = kalman.get_latest_state()
    crisis_threshold = latest["crisis_threshold"]

    # ── Reward module ──
    reward_mod = RewardModule()

    env = MacroPolicyEnvV2(
        transition_model=transition,
        regime_model=regime,
        historical_states=hist_states,
        eigenvalues_history=hist_eigen,
        fed_rate_history=hist_fed,
        crisis_threshold=crisis_threshold,
        reward_module=reward_mod,
        seed=settings.ppo_seed,
    )
    return env


def train(
    total_timesteps: int | None = None,
    seed: int | None = None,
    checkpoint_path: Path | str | None = None,
) -> Path:
    """
    Train PPO and save the policy.

    Returns the path to the saved checkpoint.
    """
    try:
        import torch.nn as nn
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError as e:
        raise RuntimeError(
            "stable-baselines3 and torch are required.  "
            "pip install stable-baselines3 torch"
        ) from e

    total_timesteps = total_timesteps or settings.ppo_total_timesteps
    seed = seed if seed is not None else settings.ppo_seed
    checkpoint_path = Path(checkpoint_path or settings.ppo_checkpoint_path)
    tb_log = settings.ppo_tensorboard_log

    logger.info("Building environment …")
    env = _build_env()

    # Map activation name → torch class
    act_map = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "GELU": nn.GELU}
    activation_fn = act_map.get(settings.ppo_activation, nn.Tanh)

    policy_kwargs = dict(
        net_arch=settings.ppo_net_arch,
        activation_fn=activation_fn,
    )

    logger.info(
        f"Initialising PPO — timesteps={total_timesteps}, "
        f"arch={settings.ppo_net_arch}, lr={settings.ppo_lr}"
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=settings.ppo_n_steps,
        batch_size=settings.ppo_batch_size,
        gamma=settings.ppo_gamma,
        gae_lambda=settings.ppo_gae_lambda,
        learning_rate=settings.ppo_lr,
        clip_range=settings.ppo_clip_range,
        ent_coef=settings.ppo_ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
        tensorboard_log=tb_log,
    )

    logger.info("Training …")
    model.learn(total_timesteps=total_timesteps)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_path).removesuffix(".zip"))
    logger.info(f"PPO policy saved → {checkpoint_path}")

    env.close()
    return checkpoint_path


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PPO macro-policy agent")
    parser.add_argument("--steps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Checkpoint path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train(total_timesteps=args.steps, seed=args.seed, checkpoint_path=args.out)


if __name__ == "__main__":
    main()
