"""
Gymnasium environment for inflation-aware Fed monetary-policy control.

State vector (dim 12):
    [ x_stress, x_liquidity, x_growth, x_inflation_gap,  (4)  state
      regime_normal, regime_fragile, regime_crisis,       (3)  one-hot
      last_action,                                        (1)  scalar
      current_fed_rate,                                   (1)  level
      eigen_1, eigen_2, eigen_3 ]                         (3)  top eigenvalues

Action:
    Continuous Box(−1, +1) → scaled to ±max_rate_step (% points)

Constraints:
    - Zero Lower Bound (ZLB): fed_rate ≥ 0
    - Max rate step per period (configurable, default 0.5 %)
    - Policy inertia penalty in reward

Reward:
    Delegated to ``RewardModule``:
    −(w₁·stress² + w₂·inflation_gap² + w₃·crisis + w₄·Δrate² + w₅·taylor²)
"""

import logging
from typing import Optional

import numpy as np

from app.config import settings
from app.environment.reward import RewardModule
from app.models.base_transition import BaseStateTransition

logger = logging.getLogger(__name__)

# ── Lazy gymnasium import ──────────────────────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("gymnasium not installed — MacroPolicyEnvV2 unavailable.")

    class _StubEnv:
        pass

    class _StubSpaces:
        class Box:
            def __init__(self, *a, **kw):
                pass

    class gym:  # type: ignore[no-redef]
        Env = _StubEnv  # type: ignore[assignment]

    spaces = _StubSpaces  # type: ignore[assignment]


class MacroPolicyEnvV2(gym.Env):  # type: ignore[misc]
    """
    Gymnasium environment for inflation-aware monetary-policy RL.

    Parameters
    ----------
    transition_model : BaseStateTransition
        Linear or neural model predicting next 4-dim state.
    regime_model : RegimeModel
        Markov regime-switching dynamics.
    historical_states : np.ndarray, shape (N, 4)
        Pool of historical 4-dim states for episode resets.
    eigenvalues_history : np.ndarray, shape (N, 3), optional
        Top-3 eigenvalues aligned with historical states.
    fed_rate_history : np.ndarray, shape (N,), optional
        Historical federal funds rate aligned with states.
    crisis_threshold : float
        Stress level above which we count a "crisis".
    reward_module : RewardModule, optional
        Custom reward module (defaults to config-based).
    max_rate_step : float, optional
        Maximum absolute rate change per step in % points.
    episode_length : int, optional
        Steps per episode.
    seed : int
        RNG seed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transition_model: BaseStateTransition,
        regime_model: "RegimeModel",
        historical_states: np.ndarray,
        eigenvalues_history: Optional[np.ndarray] = None,
        fed_rate_history: Optional[np.ndarray] = None,
        crisis_threshold: float = 2.0,
        reward_module: Optional[RewardModule] = None,
        max_rate_step: Optional[float] = None,
        episode_length: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()

        self.transition = transition_model
        self.regime = regime_model
        self.hist_states = historical_states.astype(np.float32)
        self.hist_eigen = eigenvalues_history
        self.hist_fed_rate = fed_rate_history
        self.crisis_threshold = crisis_threshold
        self.reward_mod = reward_module or RewardModule()

        self.max_rate_step = max_rate_step or settings.gym_max_rate_step
        self.ep_length = episode_length or settings.gym_episode_length
        self.zlb = settings.gym_zero_lower_bound

        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        # ── Spaces ──
        obs_dim = settings.gym_state_dim  # 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ── Internal state ──
        self._x: np.ndarray = np.zeros(4, dtype=np.float32)  # [stress, liq, growth, infl_gap]
        self._regime: int = 0
        self._last_action: float = 0.0
        self._fed_rate: float = 0.03  # initial 3 %
        self._eigenvalues: np.ndarray = np.zeros(3, dtype=np.float32)

    # ── observation ───────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        """Construct 12-dim observation vector."""
        regime_oh = np.zeros(3, dtype=np.float32)
        regime_oh[self._regime] = 1.0
        obs = np.concatenate([
            self._x.astype(np.float32),                          # 4
            regime_oh,                                             # 3
            np.array([self._last_action], dtype=np.float32),      # 1
            np.array([self._fed_rate], dtype=np.float32),         # 1
            self._eigenvalues[:3].astype(np.float32),             # 3
        ])
        return obs  # total = 12

    # ── Gym API ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        idx = self._rng.integers(0, len(self.hist_states))

        # Sample historical 4-dim state
        state = self.hist_states[idx].copy()
        if state.shape[0] == 3:
            state = np.append(state, 0.0)  # no inflation gap data
        self._x = state.astype(np.float32)

        self._regime = self.regime.initial_regime(float(self._x[0]))
        self._last_action = 0.0
        self._step_count = 0

        # Fed rate from history or default
        if self.hist_fed_rate is not None and idx < len(self.hist_fed_rate):
            self._fed_rate = float(self.hist_fed_rate[idx])
        else:
            self._fed_rate = 0.03

        # Eigenvalues
        if self.hist_eigen is not None and idx < len(self.hist_eigen):
            self._eigenvalues = self.hist_eigen[idx].copy().astype(np.float32)
        else:
            self._eigenvalues = np.zeros(3, dtype=np.float32)

        info: dict = {"regime": self._regime, "fed_rate": self._fed_rate}
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
        delta_rate = action_clipped * self.max_rate_step  # Δ in % points

        # ── Apply ZLB constraint ──
        new_rate = self._fed_rate + delta_rate
        if self.zlb and new_rate < 0.0:
            delta_rate = -self._fed_rate  # clamp so rate = 0
            new_rate = 0.0
            action_clipped = delta_rate / self.max_rate_step if self.max_rate_step > 0 else 0.0
        self._fed_rate = new_rate

        # 1. State transition (deterministic)
        x_next_det = self.transition.predict(self._x, action=delta_rate, regime=self._regime)

        # 2. Regime transition
        self._regime = self.regime.sample_next_regime(self._regime, rng=self._rng)

        # 3. Regime-scaled noise
        noise_scale = self.regime.noise_scale(self._regime)
        noise = self._rng.normal(0, 0.05 * noise_scale, size=x_next_det.shape).astype(np.float32)
        self._x = (x_next_det + noise).astype(np.float32)

        # 4. Reward via RewardModule
        stress = float(self._x[0])
        inflation_gap = float(self._x[3]) if self._x.shape[0] > 3 else 0.0
        growth = float(self._x[2]) if self._x.shape[0] > 2 else 0.0
        crisis_prob = 1.0 if stress > self.crisis_threshold else 0.0

        rc = self.reward_mod.compute(
            stress=stress,
            inflation=inflation_gap,
            crisis_prob=crisis_prob,
            rate_change=delta_rate,
            current_rate=self._fed_rate,
            growth=growth,
        )
        reward = rc.total

        self._last_action = action_clipped
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.ep_length

        info = {
            "regime": self._regime,
            "stress": stress,
            "inflation_gap": inflation_gap,
            "fed_rate": self._fed_rate,
            "delta_rate": delta_rate,
            "delta_bps": delta_rate * 100,  # backward compat
            "crisis": crisis_prob > 0.5,
            "reward_components": {
                "stress": rc.stress_penalty,
                "inflation": rc.inflation_penalty,
                "crisis": rc.crisis_penalty,
                "rate_change": rc.rate_change_penalty,
                "taylor": rc.taylor_penalty,
            },
        }

        return self._obs(), reward, terminated, truncated, info
