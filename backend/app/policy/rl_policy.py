"""
RL Policy Engine — inference wrapper for the trained PPO agent.

Provides a clean interface that mirrors the heuristic policy engine
so the backend can swap between modes transparently.

Usage:
    engine = RLPolicyEngine()
    engine.load_model()
    result = engine.predict(state_obs)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy import of SB3 (graceful fallback)
try:
    from stable_baselines3 import PPO

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None  # type: ignore[assignment, misc]
    logger.warning("stable-baselines3 not installed — RL policy unavailable.")


class RLPolicyEngine:
    """
    Wraps a trained stable-baselines3 PPO model for inference.

    Attributes:
        model: Loaded PPO model (or None before ``load_model``).
        is_loaded: Whether a model has been loaded successfully.
    """

    def __init__(self, checkpoint_path: Path | str | None = None):
        self.checkpoint_path = Path(
            checkpoint_path or settings.ppo_checkpoint_path
        )
        self.model: Optional["PPO"] = None
        self.is_loaded: bool = False

    @property
    def is_available(self) -> bool:
        """True if the SB3 library is installed AND a checkpoint exists."""
        return SB3_AVAILABLE and self.checkpoint_path.exists()

    def load_model(self) -> None:
        """Load the saved PPO policy from disk."""
        if not SB3_AVAILABLE:
            raise RuntimeError(
                "stable-baselines3 is required.  pip install stable-baselines3"
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"PPO checkpoint not found: {self.checkpoint_path}. "
                "Train first with:  python -m app.rl.train_ppo"
            )
        self.model = PPO.load(str(self.checkpoint_path))
        self.is_loaded = True
        logger.info(f"RL policy loaded from {self.checkpoint_path}")

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> dict:
        """
        Run a single inference step.

        Args:
            obs: Observation vector matching ``MacroPolicyEnvV2.observation_space``
                 (dim 12: state(4) + regime_oh(3) + last_action(1) + fed_rate(1) + eigen(3)).
            deterministic: If True, use the mode of the policy distribution.

        Returns:
            dict with keys:
                action_raw   — raw network output in [−1, +1]
                delta_rate   — mapped to % point change
                delta_bps    — mapped to basis-point change (backward compat)
                action_label — human-readable label
        """
        if not self.is_loaded:
            self.load_model()

        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]

        action, _states = self.model.predict(obs, deterministic=deterministic)
        action_val = float(action[0][0]) if action.ndim > 1 else float(action[0])
        delta_rate = action_val * settings.gym_max_rate_step  # % points
        delta_bps = delta_rate * 100

        # Label
        if delta_bps < -10:
            label = "Ease"
        elif delta_bps > 10:
            label = "Tighten"
        else:
            label = "Hold"

        return {
            "action_raw": action_val,
            "delta_rate": round(delta_rate, 4),
            "delta_bps": round(delta_bps, 1),
            "action_label": label,
        }

    def predict_from_state(
        self,
        x_t: np.ndarray,
        regime: int = 0,
        last_action: float = 0.0,
        fed_rate: float = 0.03,
        eigenvalues: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> dict:
        """
        Convenience wrapper: build the 12-dim observation from components.

        Args:
            x_t: state vector (3 or 4 dims)
            regime: current regime {0, 1, 2}
            last_action: previous raw action in [−1, +1]
            fed_rate: current federal funds rate (fraction)
            eigenvalues: top-3 eigenvalues, or None → zeros
            deterministic: use policy mode

        Returns:
            Same dict as ``predict``.
        """
        x = np.asarray(x_t, dtype=np.float32).ravel()
        if x.shape[0] == 3:
            x = np.append(x, 0.0)  # pad inflation_gap

        regime_oh = np.zeros(3, dtype=np.float32)
        regime_oh[regime] = 1.0

        eigen = np.zeros(3, dtype=np.float32)
        if eigenvalues is not None:
            eigen[:len(eigenvalues)] = eigenvalues[:3]

        obs = np.concatenate([
            x,                                                     # 4
            regime_oh,                                              # 3
            np.array([last_action], dtype=np.float32),             # 1
            np.array([fed_rate], dtype=np.float32),                # 1
            eigen,                                                  # 3
        ])
        return self.predict(obs, deterministic=deterministic)
