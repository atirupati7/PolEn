"""
Abstract base class for state transition models.

All transition models implement:
    predict(state, action) → next_state

Concrete implementations:
    - LinearStateTransition   (Ax + Ba + ε)
    - NeuralStateTransition   (MLP + optional linear residual)
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseStateTransition(ABC):
    """
    Abstract base for macro-state transition dynamics.

    State vector: x ∈ ℝ^state_dim  (typically 4: stress, liquidity, growth, inflation_gap)
    Action:       a ∈ ℝ             (Δ federal funds rate in % points)
    """

    @abstractmethod
    def predict(
        self,
        x_t: np.ndarray,
        action: float,
        regime: int = 0,
    ) -> np.ndarray:
        """
        Predict the deterministic component of the next state.

        Args:
            x_t:    current state vector, shape (state_dim,)
            action: policy action (Δ rate in % points)
            regime: current macro regime {0=Normal, 1=Fragile, 2=Crisis}

        Returns:
            x_{t+1}: predicted next state, shape (state_dim,)
        """
        ...

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the state vector."""
        ...
