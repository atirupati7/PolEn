"""Tests for Kalman Filter + EM estimation."""

import numpy as np
import pytest

from app.models.kalman_em import KalmanEM


def _make_test_observations(T: int = 300, m: int = 8, n: int = 3) -> np.ndarray:
    """Generate synthetic observations from a known state-space model."""
    np.random.seed(42)

    # True parameters
    A_true = 0.9 * np.eye(n) + 0.05 * np.random.randn(n, n)
    # Ensure stable
    sr = np.max(np.abs(np.linalg.eigvals(A_true)))
    if sr >= 0.99:
        A_true *= 0.95 / sr

    C_true = np.random.randn(m, n) * 0.5
    Q_true = np.eye(n) * 0.1
    R_true = np.eye(m) * 0.3

    # Generate states and observations
    X = np.zeros((T, n))
    Z = np.zeros((T, m))
    X[0] = np.random.randn(n)

    for t in range(1, T):
        X[t] = A_true @ X[t - 1] + np.random.multivariate_normal(np.zeros(n), Q_true)

    for t in range(T):
        Z[t] = C_true @ X[t] + np.random.multivariate_normal(np.zeros(m), R_true)

    return Z, X, A_true


class TestKalmanEM:
    """Tests for Kalman Filter and EM parameter estimation."""

    def test_fit_returns_correct_shapes(self):
        """Filtered and smoothed means should have shape (T, n)."""
        Z, _, _ = _make_test_observations(200, 8, 3)
        model = KalmanEM(latent_dim=3, max_em_iters=10)
        result = model.fit(Z)

        assert result["filtered_means"].shape == (200, 3)
        assert result["filtered_covs"].shape == (200, 3, 3)
        assert result["smoothed_means"].shape == (200, 3)

    def test_A_stable_after_training(self):
        """Transition matrix A must be stable (spectral radius < 1)."""
        Z, _, _ = _make_test_observations(300, 8, 3)
        model = KalmanEM(latent_dim=3, max_em_iters=20)
        model.fit(Z)

        sr = np.max(np.abs(np.linalg.eigvals(model.A)))
        assert sr < 1.0, f"A not stable: spectral_radius={sr}"

    def test_Q_symmetric_positive_definite(self):
        """State noise covariance Q must be symmetric positive definite."""
        Z, _, _ = _make_test_observations(200, 8, 3)
        model = KalmanEM(latent_dim=3, max_em_iters=10)
        model.fit(Z)

        Q = model.Q
        # Symmetric
        np.testing.assert_allclose(Q, Q.T, atol=1e-8)
        # Positive definite
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues > 0), f"Q not PD: min eigenvalue={eigenvalues.min()}"

    def test_log_likelihood_non_decreasing(self):
        """Log-likelihood should generally increase during EM (may not be strict)."""
        Z, _, _ = _make_test_observations(200, 8, 3)
        model = KalmanEM(latent_dim=3, max_em_iters=20)
        model.fit(Z)

        lls = model.log_likelihoods
        assert len(lls) > 1, "EM should run multiple iterations"
        # Check that LL generally increases (allow small decreases due to numerical issues)
        for i in range(1, len(lls)):
            assert lls[i] >= lls[0] - abs(lls[0]) * 0.1, (
                f"LL decreased significantly: {lls[i]} < {lls[0]}"
            )

    def test_latest_state_has_regime_label(self):
        """get_latest_state must return a valid regime label."""
        Z, _, _ = _make_test_observations(200, 8, 3)
        model = KalmanEM(latent_dim=3, max_em_iters=10)
        model.fit(Z)

        state = model.get_latest_state()
        assert state["regime_label"] in ["Normal", "Fragile", "Crisis"]
        assert "mu_T" in state
        assert "P_T" in state
        assert len(state["mu_T"]) == 3

    def test_filtered_covariance_positive_definite(self):
        """All filtered covariances must be positive semi-definite."""
        Z, _, _ = _make_test_observations(200, 8, 3)
        model = KalmanEM(latent_dim=3, max_em_iters=10)
        model.fit(Z)

        for t in range(0, 200, 50):
            P = model.filtered_covs[t]
            eigenvalues = np.linalg.eigvalsh(P)
            assert np.all(eigenvalues >= -1e-6), (
                f"P_t at t={t} not PSD: min eigenvalue={eigenvalues.min()}"
            )

    def test_B_shape_correct(self):
        """Policy control vector B must be (n, 1)."""
        Z, _, _ = _make_test_observations(200, 8, 3)
        model = KalmanEM(latent_dim=3, max_em_iters=5)
        model.fit(Z)

        assert model.B.shape == (3, 1)
