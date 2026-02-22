"""
Gymnasium environment for macroeconomic monetary-policy control.

Wraps the neural transition model, regime switching dynamics, and a
configurable reward function into a standard ``gymnasium.Env`` so that
any compatible RL algorithm (PPO, SAC, …) can be trained on it.

State vector (dim ≈ 10):
    [ x_stress, x_liquidity, x_growth,        (3)  latent state
      regime_normal, regime_fragile, regime_crisis,  (3)  one-hot
      last_action,                                   (1)  scalar
      eigen_1, eigen_2, eigen_3 ]                    (3)  top eigenvalues

Action:
    Continuous Box(−1, +1) → scaled to ±max_rate_step_bps

Reward:
    −(w1·stress² + w2·crisis_prob + w3·tail_risk + w4·action²)
"""

import logging
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ── Lazy gymnasium import ──────────────────────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("gymnasium not installed — MacroPolicyEnv unavailable.")

    # Provide stub so the file can be imported without crashing.
    class _StubEnv:
        pass

    class _StubSpaces:
        class Box:
            def __init__(self, *a, **kw):
                pass

    class gym:  # type: ignore[no-redef]
        Env = _StubEnv  # type: ignore[assignment]

    spaces = _StubSpaces  # type: ignore[assignment]


class MacroPolicyEnv(gym.Env):  # type: ignore[misc]
    """
    Gymnasium environment for monetary-policy RL.

    Parameters
    ----------
    neural_model : NeuralTransitionModel
        Trained MLP that predicts next latent state.
    regime_model : RegimeModel
        Markov regime-switching dynamics.
    historical_states : np.ndarray, shape (N, 3)
        Pool of historical latent states for episode resets.
    eigenvalues_history : np.ndarray, shape (N, 3), optional
        Top-3 eigenvalues aligned with historical states.
    crisis_threshold : float
        Stress level above which we count a "crisis".
    max_rate_step : float
        Maximum absolute rate change in bps.
    episode_length : int
        Steps per episode.
    reward_weights : dict, optional
        Override default reward weights.
    seed : int
        RNG seed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        neural_model: "NeuralTransitionModel",
        regime_model: "RegimeModel",
        historical_states: np.ndarray,
        eigenvalues_history: Optional[np.ndarray] = None,
        crisis_threshold: float = 2.0,
        max_rate_step: float | None = None,
        episode_length: int | None = None,
        reward_weights: dict | None = None,
        seed: int = 42,
    ):
        super().__init__()

        self.neural = neural_model
        self.regime = regime_model
        self.hist_states = historical_states
        self.hist_eigen = eigenvalues_history
        self.crisis_threshold = crisis_threshold

        self.max_rate_step = max_rate_step or settings.gym_max_rate_step_bps
        self.ep_length = episode_length or settings.gym_episode_length

        self.rw = {
            "stress": settings.gym_reward_w_stress,
            "crisis": settings.gym_reward_w_crisis,
            "tail_risk": settings.gym_reward_w_tail,
            "action_penalty": settings.gym_reward_w_action,
        }
        if reward_weights:
            self.rw.update(reward_weights)

        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        # ── Spaces ──
        # Observation: latent(3) + regime_oh(3) + last_action(1) + eigen(3) = 10
        obs_dim = settings.gym_state_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Internal state
        self._x: np.ndarray = np.zeros(3, dtype=np.float32)
        self._regime: int = 0
        self._last_action: float = 0.0
        self._eigenvalues: np.ndarray = np.zeros(3, dtype=np.float32)

    # ── helpers ────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        """Construct observation vector from internal state."""
        regime_oh = np.zeros(3, dtype=np.float32)
        regime_oh[self._regime] = 1.0
        obs = np.concatenate([
            self._x.astype(np.float32),
            regime_oh,
            np.array([self._last_action], dtype=np.float32),
            self._eigenvalues[:3].astype(np.float32),
        ])
        return obs

    def _compute_reward(self, action_raw: float) -> float:
        """
        Reward = −(w1·stress² + w2·crisis_indicator + w3·|stress| + w4·action²)
        """
        stress = float(self._x[0])
        crisis = 1.0 if stress > self.crisis_threshold else 0.0
        tail = abs(stress)  # simple proxy for tail risk
        action_sq = action_raw ** 2

        reward = -(
            self.rw["stress"] * stress ** 2
            + self.rw["crisis"] * crisis
            + self.rw["tail_risk"] * tail
            + self.rw["action_penalty"] * action_sq
        )
        return float(reward)

    # ── Gym API ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a random historical state
        idx = self._rng.integers(0, len(self.hist_states))
        self._x = self.hist_states[idx].copy().astype(np.float32)
        self._regime = self.regime.initial_regime(float(self._x[0]))
        self._last_action = 0.0
        self._step_count = 0

        if self.hist_eigen is not None and idx < len(self.hist_eigen):
            self._eigenvalues = self.hist_eigen[idx].copy().astype(np.float32)
        else:
            self._eigenvalues = np.zeros(3, dtype=np.float32)

        info: dict = {"regime": self._regime}
        return self._obs(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: array of shape (1,) in [−1, +1].

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action_clipped = float(np.clip(action[0], -1.0, 1.0))
        delta_bps = action_clipped * self.max_rate_step

        # 1. Neural transition (deterministic component)
        x_next_det = self.neural.predict(self._x, u_t=delta_bps, regime=self._regime)

        # 2. Regime transition
        self._regime = self.regime.sample_next_regime(self._regime, rng=self._rng)

        # 3. Gaussian noise scaled by regime
        noise_scale = self.regime.noise_scale(self._regime)
        noise = self._rng.normal(0, 0.05 * noise_scale, size=x_next_det.shape).astype(np.float32)
        self._x = (x_next_det + noise).astype(np.float32)

        # 4. Reward
        reward = self._compute_reward(action_clipped)

        self._last_action = action_clipped
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.ep_length

        info = {
            "regime": self._regime,
            "stress": float(self._x[0]),
            "delta_bps": delta_bps,
            "crisis": float(self._x[0]) > self.crisis_threshold,
        }

        return self._obs(), reward, terminated, truncated, info
