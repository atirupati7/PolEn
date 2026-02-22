"""
Regime switching model with Markov transition dynamics.

Defines regimes s ∈ {0, 1, 2} (Normal, Fragile, Crisis) with:
- Markov transition matrix Π
- Regime-dependent process noise scaling
"""

import numpy as np
from app.config import settings


class RegimeModel:
    """
    Markov regime switching for Monte Carlo simulation.

    Regimes:
        0 = Normal:  base noise
        1 = Fragile:  1.8x noise
        2 = Crisis:   3.0x noise
    """

    def __init__(
        self,
        transition_matrix: np.ndarray = None,
        noise_scales: np.ndarray = None,
    ):
        if transition_matrix is None:
            self.Pi = np.array(settings.default_transition_matrix)
        else:
            self.Pi = np.array(transition_matrix)

        if noise_scales is None:
            self.scales = np.array(settings.regime_noise_scales)
        else:
            self.scales = np.array(noise_scales)

        # Validate
        assert self.Pi.shape == (3, 3), "Transition matrix must be 3x3"
        assert np.allclose(self.Pi.sum(axis=1), 1.0, atol=1e-6), "Rows must sum to 1"
        assert len(self.scales) == 3, "Need 3 noise scales"

        # Precompute cumulative transition for sampling
        self.Pi_cumsum = np.cumsum(self.Pi, axis=1)

    def initial_regime(self, stress_score: float) -> int:
        """Determine initial regime from current stress score."""
        if stress_score < settings.regime_threshold_fragile:
            return 0
        elif stress_score < settings.regime_threshold_crisis:
            return 1
        else:
            return 2

    def sample_next_regime(self, current: int, rng: np.random.Generator = None) -> int:
        """Sample next regime given current state."""
        if rng is None:
            u = np.random.random()
        else:
            u = rng.random()

        cumprob = self.Pi_cumsum[current]
        for j in range(3):
            if u <= cumprob[j]:
                return j
        return 2  # fallback

    def noise_scale(self, regime: int) -> float:
        """Get noise scaling factor for a given regime."""
        return self.scales[regime]

    def get_regime_labels(self) -> list[str]:
        return ["Normal", "Fragile", "Crisis"]

    def to_dict(self) -> dict:
        return {
            "transition_matrix": self.Pi.tolist(),
            "noise_scales": self.scales.tolist(),
            "labels": self.get_regime_labels(),
        }
