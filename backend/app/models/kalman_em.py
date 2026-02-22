"""
Kalman Filter + EM parameter estimation for the latent macro state model.

State-space model:
    X_{t+1} = A X_t + B u_t + w_t,   w_t ~ N(0, Q)
    Z_t     = C X_t + v_t,            v_t ~ N(0, R)

Where:
    X_t ∈ R^3: [Stress, Liquidity, Growth] (latent)
    Z_t ∈ R^m: structural features from cross-asset analysis
    u_t: policy control input (scalar, in bps)

Implements:
    - Kalman filter (forward pass)
    - Rauch-Tung-Striebel smoother (backward pass)
    - EM algorithm for parameter estimation (A, C, Q, R; optionally B)
"""

import logging
from typing import Optional

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

from app.config import settings

logger = logging.getLogger(__name__)


class KalmanEM:
    """
    Kalman Filter + EM for latent macro state estimation.

    Estimates the parameters of the linear Gaussian state-space model
    and returns filtered/smoothed state estimates.
    """

    def __init__(
        self,
        latent_dim: int = 3,
        max_em_iters: int = 50,
        em_tol: float = 1e-4,
        stability_limit: float = 0.995,
    ):
        self.n = latent_dim  # state dimension
        self.max_em_iters = max_em_iters
        self.em_tol = em_tol
        self.stability_limit = stability_limit

        # Parameters (initialized in fit())
        self.A: Optional[np.ndarray] = None  # n x n transition
        self.B: Optional[np.ndarray] = None  # n x 1 control
        self.C: Optional[np.ndarray] = None  # m x n observation
        self.Q: Optional[np.ndarray] = None  # n x n state noise
        self.R: Optional[np.ndarray] = None  # m x m observation noise
        self.x0: Optional[np.ndarray] = None  # initial state mean
        self.P0: Optional[np.ndarray] = None  # initial state covariance

        # Results
        self.filtered_means: Optional[np.ndarray] = None
        self.filtered_covs: Optional[np.ndarray] = None
        self.smoothed_means: Optional[np.ndarray] = None
        self.smoothed_covs: Optional[np.ndarray] = None
        self.log_likelihoods: list = []

    def _initialize_params(self, Z: np.ndarray):
        """
        Initialize model parameters from data using PCA and heuristics.

        Args:
            Z: observation matrix (T x m)
        """
        T, m = Z.shape
        n = self.n

        # Use PCA to initialize C (observation matrix)
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(Z)  # T x n

        self.C = pca.components_.T  # m x n

        # Initialize A from VAR(1) on PCA scores
        X_lag = X_pca[:-1]  # (T-1) x n
        X_lead = X_pca[1:]  # (T-1) x n
        try:
            self.A = np.linalg.lstsq(X_lag, X_lead, rcond=None)[0].T  # n x n
        except np.linalg.LinAlgError:
            self.A = 0.95 * np.eye(n)

        # Enforce stability
        self._enforce_stability()

        # B: policy control vector (n x 1)
        self.B = np.array(settings.default_B).reshape(n, 1)

        # Q: state noise covariance from residuals
        residuals = X_lead - X_lag @ self.A.T
        self.Q = np.cov(residuals, rowvar=False)
        self.Q = 0.5 * (self.Q + self.Q.T)  # Ensure symmetry
        self.Q += 1e-6 * np.eye(n)  # Regularize

        # R: observation noise covariance
        Z_reconstructed = X_pca @ self.C.T
        obs_residuals = Z - Z_reconstructed
        self.R = np.diag(np.var(obs_residuals, axis=0))
        self.R += 1e-6 * np.eye(m)

        # Initial state
        self.x0 = X_pca[0]
        self.P0 = np.eye(n) * np.var(X_pca, axis=0).mean()

        logger.info(
            f"Initialized KalmanEM: A spectral_radius={self._spectral_radius():.4f}, "
            f"state_dim={n}, obs_dim={m}"
        )

    def _spectral_radius(self) -> float:
        """Compute spectral radius of A."""
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def _enforce_stability(self):
        """Scale A down if spectral radius exceeds limit."""
        sr = self._spectral_radius()
        if sr >= self.stability_limit:
            scale = self.stability_limit / (sr + 1e-10)
            self.A *= scale
            logger.info(f"Scaled A: spectral_radius {sr:.4f} -> {self._spectral_radius():.4f}")

    def kalman_filter(
        self,
        Z: np.ndarray,
        u: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Forward Kalman filter pass.

        Args:
            Z: observations (T x m)
            u: control inputs (T x 1) or None

        Returns:
            (filtered_means, filtered_covs, log_likelihood)
            filtered_means: (T x n)
            filtered_covs: (T x n x n)
        """
        T, m = Z.shape
        n = self.n
        A, C, Q, R = self.A, self.C, self.Q, self.R
        B = self.B

        # Allocate
        x_filt = np.zeros((T, n))
        P_filt = np.zeros((T, n, n))
        x_pred = np.zeros((T, n))
        P_pred = np.zeros((T, n, n))

        log_lik = 0.0

        for t in range(T):
            # Predict
            if t == 0:
                x_pred[t] = self.x0
                P_pred[t] = self.P0
            else:
                x_p = A @ x_filt[t - 1]
                if u is not None and B is not None:
                    x_p += (B @ u[t - 1].reshape(1, -1).T).flatten()
                x_pred[t] = x_p
                P_pred[t] = A @ P_filt[t - 1] @ A.T + Q

            # Innovation
            z_pred = C @ x_pred[t]
            innov = Z[t] - z_pred
            S = C @ P_pred[t] @ C.T + R

            # Regularize S
            S = 0.5 * (S + S.T)
            S += 1e-8 * np.eye(m)

            # Kalman gain
            try:
                S_inv = np.linalg.solve(S, np.eye(m))
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            K = P_pred[t] @ C.T @ S_inv

            # Update
            x_filt[t] = x_pred[t] + K @ innov
            P_filt[t] = (np.eye(n) - K @ C) @ P_pred[t]
            P_filt[t] = 0.5 * (P_filt[t] + P_filt[t].T)

            # Log-likelihood contribution
            try:
                sign, logdet = np.linalg.slogdet(S)
                if sign > 0:
                    log_lik += -0.5 * (m * np.log(2 * np.pi) + logdet + innov @ S_inv @ innov)
            except np.linalg.LinAlgError:
                pass

        return x_filt, P_filt, x_pred, P_pred, log_lik

    def rts_smoother(
        self,
        x_filt: np.ndarray,
        P_filt: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
    ) -> tuple:
        """
        Rauch-Tung-Striebel smoother (backward pass).

        Returns:
            (smoothed_means, smoothed_covs, cross_covs)
        """
        T, n = x_filt.shape
        A = self.A

        x_smooth = np.zeros_like(x_filt)
        P_smooth = np.zeros_like(P_filt)
        G = np.zeros((T - 1, n, n))  # Smoother gains

        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for t in range(T - 2, -1, -1):
            P_pred_t1 = P_pred[t + 1]
            P_pred_t1 = 0.5 * (P_pred_t1 + P_pred_t1.T) + 1e-8 * np.eye(n)

            try:
                G[t] = P_filt[t] @ A.T @ np.linalg.inv(P_pred_t1)
            except np.linalg.LinAlgError:
                G[t] = P_filt[t] @ A.T @ np.linalg.pinv(P_pred_t1)

            x_smooth[t] = x_filt[t] + G[t] @ (x_smooth[t + 1] - x_pred[t + 1])
            P_smooth[t] = P_filt[t] + G[t] @ (P_smooth[t + 1] - P_pred_t1) @ G[t].T
            P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

        # Cross-covariance E[x_t x_{t-1}^T | all data]
        cross_covs = np.zeros((T - 1, n, n))
        for t in range(T - 1):
            cross_covs[t] = G[t] @ P_smooth[t + 1]

        return x_smooth, P_smooth, cross_covs

    def em_step(
        self,
        Z: np.ndarray,
        x_smooth: np.ndarray,
        P_smooth: np.ndarray,
        cross_covs: np.ndarray,
    ):
        """
        Single EM M-step: update parameters given smoothed estimates.
        """
        T, m = Z.shape
        n = self.n

        # Sufficient statistics
        # E[x_t x_t^T]
        Exx = np.zeros((n, n))
        # E[x_t x_{t-1}^T]
        Exx_lag = np.zeros((n, n))
        # E[x_{t-1} x_{t-1}^T]
        Exx_prev = np.zeros((n, n))
        # sum of x_t z_t^T etc.
        sum_xz = np.zeros((m, n))

        for t in range(T):
            Exx_t = P_smooth[t] + np.outer(x_smooth[t], x_smooth[t])
            Exx += Exx_t
            sum_xz += np.outer(Z[t], x_smooth[t])

            if t > 0:
                Exx_lag += cross_covs[t - 1] + np.outer(x_smooth[t], x_smooth[t - 1])
                Exx_prev += P_smooth[t - 1] + np.outer(x_smooth[t - 1], x_smooth[t - 1])

        # Update A
        try:
            self.A = Exx_lag @ np.linalg.inv(Exx_prev)
        except np.linalg.LinAlgError:
            self.A = Exx_lag @ np.linalg.pinv(Exx_prev)

        self._enforce_stability()

        # Update C
        try:
            self.C = sum_xz @ np.linalg.inv(Exx)
        except np.linalg.LinAlgError:
            self.C = sum_xz @ np.linalg.pinv(Exx)

        # Update Q
        Q_new = np.zeros((n, n))
        for t in range(1, T):
            diff = x_smooth[t] - self.A @ x_smooth[t - 1]
            Q_new += P_smooth[t] + np.outer(diff, diff) - self.A @ cross_covs[t - 1].T - cross_covs[t - 1] @ self.A.T + self.A @ P_smooth[t - 1] @ self.A.T
        Q_new /= (T - 1)
        self.Q = 0.5 * (Q_new + Q_new.T) + 1e-8 * np.eye(n)

        # Update R
        R_new = np.zeros((m, m))
        for t in range(T):
            diff = Z[t] - self.C @ x_smooth[t]
            R_new += np.outer(diff, diff) + self.C @ P_smooth[t] @ self.C.T
        R_new /= T
        self.R = 0.5 * (R_new + R_new.T) + 1e-8 * np.eye(m)

        # Update initial state
        self.x0 = x_smooth[0]
        self.P0 = P_smooth[0]

    def fit(self, Z: np.ndarray, u: Optional[np.ndarray] = None) -> dict:
        """
        Fit the state-space model to observations Z using EM.

        Args:
            Z: observation matrix (T x m), the Z_history from structure features
            u: control inputs (T x 1) or None

        Returns:
            dict with fitted parameters and filtered/smoothed states
        """
        T, m = Z.shape
        logger.info(f"Fitting KalmanEM on {T} observations, {m} features")

        # Handle NaN/Inf
        Z = np.nan_to_num(Z, nan=0.0, posinf=3.0, neginf=-3.0)

        # Initialize parameters
        self._initialize_params(Z)

        self.log_likelihoods = []
        prev_ll = -np.inf

        for iteration in range(self.max_em_iters):
            # E-step: filter + smooth
            x_filt, P_filt, x_pred, P_pred, log_lik = self.kalman_filter(Z, u)
            x_smooth, P_smooth, cross_covs = self.rts_smoother(x_filt, P_filt, x_pred, P_pred)

            self.log_likelihoods.append(log_lik)

            # Check convergence
            if iteration > 0:
                rel_change = abs(log_lik - prev_ll) / (abs(prev_ll) + 1e-10)
                if rel_change < self.em_tol:
                    logger.info(f"EM converged at iteration {iteration}, LL={log_lik:.2f}")
                    break

            prev_ll = log_lik

            # M-step
            self.em_step(Z, x_smooth, P_smooth, cross_covs)

        # Final filter pass with estimated parameters
        x_filt, P_filt, x_pred, P_pred, log_lik = self.kalman_filter(Z, u)
        x_smooth, P_smooth, cross_covs = self.rts_smoother(x_filt, P_filt, x_pred, P_pred)

        # ── Sign correction for the stress dimension ──────────────
        # PCA initialisation and EM iterations can assign an arbitrary sign
        # to each latent dimension.  We need the first dimension ("stress")
        # to increase when observable stress rises.  The structural features
        # in Z are ordered so that Z[:, 0] (lambda_concentration) and most
        # other columns increase during stress.  Check correlation of
        # filtered X[0] with the column-mean of Z.  If negative, flip
        # dimension 0 via the similarity transform  D = diag(-1, 1, …, 1).
        z_mean_per_step = Z.mean(axis=1)          # (T,) average feature per month
        filtered_stress = x_filt[:, 0]              # (T,)
        corr_sign = np.corrcoef(z_mean_per_step, filtered_stress)[0, 1]

        if corr_sign < 0:
            logger.info(
                f"Flipping latent dim-0 sign (corr with obs = {corr_sign:.3f})"
            )
            n = self.n
            D = np.eye(n)
            D[0, 0] = -1.0

            # Similarity transform on parameters:  X' = D X
            #   A' = D A D,  Q' = D Q D,  C' = C D,  B' = D B
            self.A = D @ self.A @ D
            self.Q = D @ self.Q @ D
            self.C = self.C @ D
            self.B = D @ self.B
            self.x0 = D @ self.x0
            self.P0 = D @ self.P0 @ D

            # Flip states
            x_filt[:, 0]   *= -1
            x_smooth[:, 0] *= -1
            for t in range(len(P_filt)):
                P_filt[t]   = D @ P_filt[t] @ D
                P_smooth[t] = D @ P_smooth[t] @ D

        self.filtered_means = x_filt
        self.filtered_covs = P_filt
        self.smoothed_means = x_smooth
        self.smoothed_covs = P_smooth

        logger.info(
            f"EM complete: {len(self.log_likelihoods)} iterations, "
            f"final LL={self.log_likelihoods[-1]:.2f}, "
            f"A spectral_radius={self._spectral_radius():.4f}"
        )

        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "Q": self.Q,
            "R": self.R,
            "filtered_means": x_filt,
            "filtered_covs": P_filt,
            "smoothed_means": x_smooth,
            "smoothed_covs": P_smooth,
            "log_likelihoods": self.log_likelihoods,
        }

    def get_latest_state(self) -> dict:
        """Get the latest filtered state for Monte Carlo initialization."""
        if self.filtered_means is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        mu_T = self.filtered_means[-1]
        P_T = self.filtered_covs[-1]

        # Regime label based on stress score
        stress_series = self.filtered_means[:, 0]
        stress_mean = np.mean(stress_series)
        stress_std = np.std(stress_series) + 1e-10
        stress_score = (mu_T[0] - stress_mean) / stress_std

        if stress_score < settings.regime_threshold_fragile:
            regime_label = "Normal"
        elif stress_score < settings.regime_threshold_crisis:
            regime_label = "Fragile"
        else:
            regime_label = "Crisis"

        # Crisis threshold (95th percentile of historical filtered stress)
        crisis_threshold = float(np.percentile(
            stress_series, settings.crisis_threshold_percentile
        ))

        return {
            "mu_T": mu_T.tolist(),
            "P_T": P_T.tolist(),
            "stress_score": float(stress_score),
            "regime_label": regime_label,
            "crisis_threshold": crisis_threshold,
            "stress_mean": float(stress_mean),
            "stress_std": float(stress_std),
        }
