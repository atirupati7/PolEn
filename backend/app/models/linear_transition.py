"""
Linear state transition model.

    x_{t+1} = A x_t + B a_t + ε_t

Wraps the Kalman filter's A, B, Q matrices (3-dim) and extends them
to 4-dim by appending an AR(1) inflation-gap component:

    A_ext = | A_kalman   0 |      B_ext = | B_kalman |
            | 0 ... 0    ρ |               | b_infl   |

where ρ = inflation_persistence (≈ 0.95) and b_infl captures the
(small) immediate effect of rate changes on inflation expectations.
"""

import logging
from typing import Optional

import numpy as np

from app.config import settings
from app.models.base_transition import BaseStateTransition

logger = logging.getLogger(__name__)


class LinearStateTransition(BaseStateTransition):
    """
    Linear transition: x_{t+1} = A x_t + B a_t  (no noise here; caller adds noise).

    Parameters
    ----------
    A : np.ndarray, shape (3, 3) or (4, 4)
        State transition matrix from Kalman EM.  If (3, 3), it is
        automatically extended to (4, 4) with an inflation AR(1) row.
    B : np.ndarray, shape (3,) or (4,)
        Control-input vector.  If (3,), extended with ``b_inflation``.
    inflation_persistence : float
        AR(1) coefficient for inflation gap (ρ).
    b_inflation : float
        Sensitivity of inflation to the rate-change action.
        Negative: higher rates push inflation down.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        inflation_persistence: Optional[float] = None,
        b_inflation: float = -0.002,
    ) -> None:
        self._rho = inflation_persistence or settings.inflation_persistence

        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64).ravel()

        if A.shape == (3, 3):
            self._A = self._extend_A(A)
        elif A.shape == (4, 4):
            self._A = A
        else:
            raise ValueError(f"A must be (3,3) or (4,4), got {A.shape}")

        if B.shape == (3,):
            self._B = np.append(B, b_inflation)
        elif B.shape == (4,):
            self._B = B
        else:
            raise ValueError(f"B must be (3,) or (4,), got {B.shape}")

        logger.debug(
            f"LinearStateTransition created: A={self._A.shape}, B={self._B.shape}, ρ={self._rho}"
        )

    def _extend_A(self, A3: np.ndarray) -> np.ndarray:
        """Extend 3×3 A to 4×4 with AR(1) inflation row."""
        A4 = np.zeros((4, 4), dtype=np.float64)
        A4[:3, :3] = A3
        A4[3, 3] = self._rho
        return A4

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def B(self) -> np.ndarray:
        return self._B

    def predict(
        self,
        x_t: np.ndarray,
        action: float,
        regime: int = 0,
    ) -> np.ndarray:
        """
        Predict x_{t+1} = A x_t + B * action.

        The regime argument is accepted for interface compatibility
        but not used in the linear model (regime noise is added by the
        environment, not the transition).
        """
        x_t = np.asarray(x_t, dtype=np.float64).ravel()
        if x_t.shape[0] == 3:
            # Pad with zero inflation gap if caller passes 3-dim
            x_t = np.append(x_t, 0.0)

        x_next = self._A @ x_t + self._B * action
        return x_next.astype(np.float32)
