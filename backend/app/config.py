"""
Configuration for the MacroState Control Room backend.

Loads a centralized config.yaml and exposes typed Settings.
All modules import ``settings`` from here — no hardcoded values.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings


_BACKEND_ROOT = Path(__file__).parent.parent


def _load_yaml_config() -> dict[str, Any]:
    """Load config.yaml from project root (backend/)."""
    yaml_path = _BACKEND_ROOT / "config.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml_config()
_data = _yaml.get("data", {})
_transition = _yaml.get("transition", {})
_neural = _transition.get("neural", {})
_gym = _yaml.get("gym_env", {})
_ppo = _yaml.get("ppo", {})
_reward = _yaml.get("reward", {})


class Settings(BaseSettings):
    # ── Secrets / env vars ──────────────────────────────────────
    fred_api_key: str = ""

    # ── Paths ───────────────────────────────────────────────────
    data_cache_dir: Path = _BACKEND_ROOT / "data_cache"
    raw_cache_dir: Path = _BACKEND_ROOT / "data_cache" / "raw"
    processed_cache_dir: Path = _BACKEND_ROOT / "data_cache" / "processed"

    # ── FRED series (expanded: CPI + FEDFUNDS) ──────────────────
    fred_series: dict = {
        "SP500": "SP500",
        "DGS2": "DGS2",
        "DGS10": "DGS10",
        "BAA": "BAA",
        "VIXCLS": "VIXCLS",
        "DTWEXBGS": "DTWEXBGS",
        "DGS5": "DGS5",
        "CPIAUCSL": "CPIAUCSL",       # CPI All Urban Consumers
        "FEDFUNDS": "FEDFUNDS",        # Effective Federal Funds Rate
        "USREC": "USREC",             # NBER Recession Indicator (1=recession)
        "T10Y2Y": "T10Y2Y",           # 10Y-2Y spread (yield curve)
    }

    # ── Data architecture ───────────────────────────────────────
    data_start_year: int = _data.get("start_year", 1990)
    data_lookback_years: int = max(
        36, _data.get("start_year", 1990) and 36
    )  # derived from start_year → now
    data_frequency: str = _data.get("frequency", "monthly")
    data_use_daily_cache: bool = _data.get("use_daily_cache", True)
    inflation_target: float = _data.get("inflation_target", 0.02)

    # ── Rolling windows ─────────────────────────────────────────
    rolling_zscore_window: int = 252
    rolling_zscore_min_periods: int = 200
    rolling_cov_window: int = 60

    # ── State space ─────────────────────────────────────────────
    kalman_latent_dim: int = 3  # Kalman estimates [stress, liquidity, growth]
    state_dim: int = 4          # Full state includes inflation: [stress, liquidity, growth, inflation_gap]

    # ── Regime thresholds for UI label ──────────────────────────
    regime_threshold_fragile: float = 0.5
    regime_threshold_crisis: float = 1.5

    # ── Default policy vector B (Kalman 3-dim) ──────────────────
    default_B: list = [0.003, 0.006, -0.004]

    # ── Default regime transition matrix (3×3 row-stochastic) ───
    default_transition_matrix: list = [
        [0.92, 0.07, 0.01],
        [0.05, 0.88, 0.07],
        [0.02, 0.08, 0.90],
    ]
    regime_noise_scales: list = [1.0, 1.8, 3.0]
    # Regime-dependent A-matrix decay: A_regime = A * scale
    regime_A_scales: list = [1.0, 0.97, 0.92]
    # Student-t degrees of freedom for fat-tailed noise
    student_t_df: float = 5.0
    # Policy transmission lag (months)
    policy_lag_months: int = 3
    # Policy geometric decay factor
    policy_decay: float = 0.7
    # Indirect stress-from-growth coupling
    stress_growth_coupling: float = 0.15

    # ── Monte Carlo defaults ────────────────────────────────────
    mc_default_paths: int = 5000
    mc_default_horizon: int = 24
    mc_default_speed_ms: int = 120
    mc_spaghetti_count: int = 30
    trading_days_per_month: int = 21

    # ── Path jitter (x-direction noise for heuristic + RL) ──────
    # Adds random temporal jitter to agent paths so they look like
    # independent simulations rather than smooth copies of history.
    # Value is the maximum fractional time-shift per step (0.0 = off,
    # 0.3 = up to ±30% of one step interpolated).  Loss is preserved
    # because jitter is mean-zero and applied after metrics are computed.
    path_jitter_scale: float = 1.0

    # ── Crisis threshold ────────────────────────────────────────
    crisis_threshold_percentile: float = 95.0

    # ── Shock constants ─────────────────────────────────────────
    shock_credit_stress: float = 0.8
    shock_credit_liquidity: float = 0.5
    shock_vol_stress: float = 1.0
    shock_rate_bps: float = 50.0

    # ── Policy engine mode ──────────────────────────────────────
    policy_mode: str = _yaml.get("policy_mode", "heuristic")

    # ── Transition model ────────────────────────────────────────
    transition_model_type: str = _transition.get("model_type", "linear")
    inflation_persistence: float = _transition.get("inflation_persistence", 0.95)

    # Neural transition sub-config
    neural_transition_enabled: bool = _transition.get("model_type", "linear") == "neural"
    neural_hidden_dim: int = _neural.get("hidden_dim", 64)
    neural_n_layers: int = _neural.get("n_layers", 2)
    neural_use_layer_norm: bool = _neural.get("use_layer_norm", True)
    neural_lr: float = _neural.get("learning_rate", 1e-3)
    neural_weight_decay: float = _neural.get("weight_decay", 1e-4)
    neural_epochs: int = _neural.get("epochs", 100)
    neural_early_stopping_patience: int = _neural.get("early_stopping_patience", 10)
    neural_train_split: float = _neural.get("train_split", 0.8)
    neural_seed: int = _neural.get("seed", 42)
    neural_checkpoint_path: Path = _BACKEND_ROOT / _neural.get(
        "checkpoint_path", "models/checkpoints/neural_transition.pt"
    )

    # ── Gym environment ─────────────────────────────────────────
    gym_episode_length: int = _gym.get("episode_length", 60)
    gym_max_rate_step: float = _gym.get("max_rate_step", 0.5)  # % points
    gym_zero_lower_bound: bool = _gym.get("zero_lower_bound", True)
    gym_state_dim: int = _gym.get("state_dim", 12)

    # Backward compat alias (old code used bps)
    @property
    def gym_max_rate_step_bps(self) -> float:
        return self.gym_max_rate_step * 100  # 0.5 % → 50 bps

    # ── Reward function ─────────────────────────────────────────
    reward_w_stress: float = _reward.get("w_stress", 1.0)
    reward_w_inflation: float = _reward.get("w_inflation", 1.0)
    reward_w_crisis: float = _reward.get("w_crisis", 2.0)
    reward_w_rate_change: float = _reward.get("w_rate_change", 0.1)
    reward_w_taylor: float = _reward.get("w_taylor", 0.0)
    reward_inflation_target: float = _reward.get("inflation_target", 0.02)

    # ── PPO training ────────────────────────────────────────────
    ppo_total_timesteps: int = _ppo.get("total_timesteps", 500_000)
    ppo_n_steps: int = _ppo.get("n_steps", 2048)
    ppo_batch_size: int = _ppo.get("batch_size", 64)
    ppo_gamma: float = _ppo.get("gamma", 0.99)
    ppo_gae_lambda: float = _ppo.get("gae_lambda", 0.95)
    ppo_lr: float = _ppo.get("learning_rate", 3e-4)
    ppo_clip_range: float = _ppo.get("clip_range", 0.2)
    ppo_ent_coef: float = _ppo.get("ent_coef", 0.01)
    ppo_n_epochs: int = _ppo.get("n_epochs", 10)
    ppo_net_arch: list = _ppo.get("policy_kwargs", {}).get("net_arch", [64, 64])
    ppo_activation: str = _ppo.get("policy_kwargs", {}).get("activation_fn", "Tanh")
    ppo_seed: int = _ppo.get("seed", 42)
    ppo_checkpoint_path: Path = _BACKEND_ROOT / _ppo.get(
        "checkpoint_path", "models/checkpoints/ppo_policy.zip"
    )
    ppo_tensorboard_log: str = str(
        _BACKEND_ROOT / _ppo.get("tensorboard_log", "runs/ppo_macro")
    )

    # Backward compat aliases
    @property
    def latent_dim(self) -> int:
        """Alias for kalman_latent_dim (used by Kalman, MC, structure)."""
        return self.kalman_latent_dim

    class Config:
        env_prefix = ""
        env_file = str(_BACKEND_ROOT / ".env")
        env_file_encoding = "utf-8"


settings = Settings()

# ── Ensure directories exist ────────────────────────────────────
settings.raw_cache_dir.mkdir(parents=True, exist_ok=True)
settings.processed_cache_dir.mkdir(parents=True, exist_ok=True)
settings.neural_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
settings.ppo_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
