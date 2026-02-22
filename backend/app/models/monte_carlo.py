"""
Monte Carlo simulation engine with Numba acceleration.

Simulates forward paths of the latent macro state under policy actions,
regime switching, and shock injections. Computes fan chart percentiles,
crisis probabilities, and tail risk metrics step by step for live streaming.

Key design decisions:
- Monthly timestep (21 trading days)
- A_monthly ≈ A_daily^21 (matrix power approximation)
- Q_monthly ≈ Q_daily * 21 (noise variance scaling)
- Numba @njit for inner loop; fallback to numpy if Numba unavailable
"""

import logging
from typing import Optional

import numpy as np
from scipy import linalg as la

from app.config import settings
from app.models.regime import RegimeModel

logger = logging.getLogger(__name__)

# Try Numba; fall back gracefully
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available; using pure numpy Monte Carlo (slower)")

    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args):
        return range(*args)


def daily_to_monthly_params(
    A_daily: np.ndarray,
    Q_daily: np.ndarray,
    B_daily: np.ndarray,
    days_per_month: int = 21,
) -> tuple:
    """
    Convert daily state-space parameters to monthly frequency.

    A_monthly = A_daily^(days_per_month) via eigendecomposition for stability.
    Q_monthly ≈ sum_{k=0}^{d-1} A^k Q (A^k)^T, approximated as Q_daily * days for simplicity.
    B_monthly = sum_{k=0}^{d-1} A^k B ≈ (I - A^d)(I - A)^{-1} B

    Args:
        A_daily: (n x n) transition matrix
        Q_daily: (n x n) state noise covariance
        B_daily: (n x 1) control input
        days_per_month: trading days per month

    Returns:
        (A_monthly, Q_monthly, B_monthly)
    """
    n = A_daily.shape[0]
    d = days_per_month

    # ── Enforce stationarity on A_daily BEFORE powering ──
    eig_daily = np.linalg.eigvals(A_daily)
    spectral_radius_daily = float(np.max(np.abs(eig_daily)))
    if spectral_radius_daily >= 1.0:
        rescale = 1.0 / (1.05 * spectral_radius_daily)
        A_daily = A_daily * rescale
        logger.warning(
            f"[Stationarity] A_daily spectral radius {spectral_radius_daily:.4f} >= 1.0 "
            f"→ rescaled by {rescale:.4f} to {float(np.max(np.abs(np.linalg.eigvals(A_daily)))):.4f}"
        )
    else:
        logger.info(f"[Stationarity] A_daily spectral radius {spectral_radius_daily:.4f} — OK")

    # A_monthly = A^d via eigendecomposition
    try:
        eigenvalues, V = la.eig(A_daily)
        # Power eigenvalues
        lambda_d = eigenvalues ** d
        A_monthly = np.real(V @ np.diag(lambda_d) @ la.inv(V))
    except (la.LinAlgError, ValueError):
        # Fallback: iterative matrix power
        A_monthly = np.linalg.matrix_power(A_daily, d)

    # ── Verify monthly stationarity ──
    eig_monthly = np.linalg.eigvals(A_monthly)
    sr_monthly = float(np.max(np.abs(eig_monthly)))
    if sr_monthly >= 1.0:
        rescale_m = 1.0 / (1.05 * sr_monthly)
        A_monthly = A_monthly * rescale_m
        logger.warning(
            f"[Stationarity] A_monthly spectral radius {sr_monthly:.4f} >= 1.0 "
            f"→ rescaled to {float(np.max(np.abs(np.linalg.eigvals(A_monthly)))):.4f}"
        )
    else:
        logger.info(f"[Stationarity] A_monthly spectral radius {sr_monthly:.4f} — OK")

    # Q_monthly: geometric series of noise accumulation
    # Q_monthly = sum_{k=0}^{d-1} A^k Q (A^k)^T
    # Approximation for near-identity A: Q_monthly ≈ d * Q
    # Better: use the exact formula via vec() and Kronecker, but approximate for speed
    Q_monthly = np.zeros((n, n))
    A_power = np.eye(n)
    for k in range(d):
        Q_monthly += A_power @ Q_daily @ A_power.T
        A_power = A_power @ A_daily

    Q_monthly = 0.5 * (Q_monthly + Q_monthly.T)  # Ensure symmetry
    Q_monthly += 1e-10 * np.eye(n)

    # B_monthly: sum_{k=0}^{d-1} A^k B
    B_monthly = np.zeros_like(B_daily)
    A_power = np.eye(n)
    for k in range(d):
        B_monthly += A_power @ B_daily
        A_power = A_power @ A_daily

    return A_monthly, Q_monthly, B_monthly


@njit(cache=True)
def _simulate_paths_numba(
    A: np.ndarray,
    B: np.ndarray,
    Q_chol: np.ndarray,
    mu0: np.ndarray,
    P0_chol: np.ndarray,
    delta_bps: float,
    N: int,
    H: int,
    regime_switching: bool,
    Pi_cumsum: np.ndarray,
    noise_scales: np.ndarray,
    initial_regime: int,
    crisis_threshold: float,
    seed: int,
    regime_A_scales: np.ndarray,
    policy_lag: int,
    policy_decay: float,
    stress_growth_coupling: float,
    student_t_df: float,
) -> tuple:
    """
    Core Monte Carlo simulation loop (Numba JIT compiled).

    Fixes applied:
    - Mean-reverting transition (no drift bias)
    - Delayed policy transmission (growth at lag, stress indirect)
    - Regime-dependent A scaling
    - Student-t noise (fat tails)
    - Crisis probability = stress > threshold OR regime == Crisis
    - Empirical ES95 (mean of stress >= VaR95)
    """
    n = A.shape[0]
    np.random.seed(seed)

    # Output arrays
    stress_p5 = np.zeros(H)
    stress_p25 = np.zeros(H)
    stress_p50 = np.zeros(H)
    stress_p75 = np.zeros(H)
    stress_p95 = np.zeros(H)

    growth_p5 = np.zeros(H)
    growth_p25 = np.zeros(H)
    growth_p50 = np.zeros(H)
    growth_p75 = np.zeros(H)
    growth_p95 = np.zeros(H)

    crisis_prob = np.zeros(H)
    es95_stress = np.zeros(H)

    # Spaghetti paths (first 30)
    n_spag = min(30, N)
    spaghetti = np.zeros((H, n_spag))

    # Policy effect: delayed and decaying
    # Build a policy impulse array over horizon
    policy_impulse = np.zeros(H)
    for t in range(H):
        lag_t = t - policy_lag
        if lag_t >= 0:
            policy_impulse[t] = delta_bps * (policy_decay ** lag_t)

    # Initialize N paths from prior
    X = np.zeros((N, n))
    regimes = np.full(N, initial_regime, dtype=np.int64)

    for i in range(N):
        z = np.random.randn(n)
        for j in range(n):
            X[i, j] = mu0[j]
            for k in range(n):
                X[i, j] += P0_chol[j, k] * z[k]

    # Simulate
    stress_vals = np.zeros(N)
    growth_vals = np.zeros(N)

    for step in range(H):
        for i in range(N):
            # 1) Regime transition FIRST (before state update)
            if regime_switching:
                u_rand = np.random.random()
                curr = regimes[i]
                new_regime = 2
                for j in range(3):
                    if u_rand <= Pi_cumsum[curr, j]:
                        new_regime = j
                        break
                regimes[i] = new_regime

            # 2) Regime-dependent A scaling
            a_scale = regime_A_scales[regimes[i]] if regime_switching else 1.0

            # 3) Noise scaling (regime-dependent variance)
            noise_scale = noise_scales[regimes[i]] if regime_switching else 1.0

            # 4) Student-t noise for fat tails
            # Approximate: normal * sqrt(df / chi2(df))
            z_norm = np.random.randn(n)
            # chi2 with df degrees of freedom = sum of df standard normals squared
            chi2_val = 0.0
            for _k in range(int(student_t_df)):
                _zz = np.random.randn()
                chi2_val += _zz * _zz
            chi2_val = max(chi2_val, 0.01)  # avoid division by zero
            t_scale = np.sqrt(student_t_df / chi2_val)

            noise = np.zeros(n)
            for j in range(n):
                for k in range(n):
                    noise[j] += Q_chol[j, k] * z_norm[k]
                noise[j] *= t_scale * np.sqrt(noise_scale)

            # 5) Mean-reverting transition: X_{t+1} = A_regime * X_t + noise
            #    NO constant Bu offset — policy enters through growth channel only
            new_x = np.zeros(n)
            for j in range(n):
                for k in range(n):
                    new_x[j] += (A[j, k] * a_scale) * X[i, k]
                new_x[j] += noise[j]

            # 6) Delayed policy effect on GROWTH (index 2) only
            if n > 2:
                new_x[2] += B[2, 0] * policy_impulse[step]

            # 7) Indirect stress effect from growth
            #    stress_{t+1} += coupling * (-growth_t)
            if n > 2:
                new_x[0] += stress_growth_coupling * (-X[i, 2])

            for j in range(n):
                X[i, j] = new_x[j]

        # Collect statistics
        for i in range(N):
            stress_vals[i] = X[i, 0]
            growth_vals[i] = X[i, 2] if n > 2 else 0.0

        # Sort for percentiles
        stress_sorted = np.sort(stress_vals)
        growth_sorted = np.sort(growth_vals)

        idx_5 = int(0.05 * N)
        idx_25 = int(0.25 * N)
        idx_50 = int(0.50 * N)
        idx_75 = int(0.75 * N)
        idx_95 = int(0.95 * N)

        stress_p5[step] = stress_sorted[idx_5]
        stress_p25[step] = stress_sorted[idx_25]
        stress_p50[step] = stress_sorted[idx_50]
        stress_p75[step] = stress_sorted[idx_75]
        stress_p95[step] = stress_sorted[idx_95]

        growth_p5[step] = growth_sorted[idx_5]
        growth_p25[step] = growth_sorted[idx_25]
        growth_p50[step] = growth_sorted[idx_50]
        growth_p75[step] = growth_sorted[idx_75]
        growth_p95[step] = growth_sorted[idx_95]

        # Crisis probability: stress > threshold OR regime == Crisis (2)
        n_crisis = 0
        for i in range(N):
            if stress_vals[i] > crisis_threshold:
                n_crisis += 1
            elif regime_switching and regimes[i] == 2:
                n_crisis += 1
        crisis_prob[step] = n_crisis / N

        # Empirical ES95: mean of stress where stress >= VaR95
        var95 = stress_sorted[idx_95]
        es_sum = 0.0
        es_count = 0
        for i in range(N):
            if stress_vals[i] >= var95:
                es_sum += stress_vals[i]
                es_count += 1
        es95_stress[step] = es_sum / max(1, es_count)

        # Spaghetti
        for i in range(n_spag):
            spaghetti[step, i] = X[i, 0]

    return (
        stress_p5, stress_p25, stress_p50, stress_p75, stress_p95,
        growth_p5, growth_p25, growth_p50, growth_p75, growth_p95,
        crisis_prob, es95_stress, spaghetti,
    )


def simulate_paths_numpy(
    A: np.ndarray,
    B: np.ndarray,
    Q_chol: np.ndarray,
    mu0: np.ndarray,
    P0_chol: np.ndarray,
    delta_bps: float,
    N: int,
    H: int,
    regime_switching: bool,
    Pi_cumsum: np.ndarray,
    noise_scales: np.ndarray,
    initial_regime: int,
    crisis_threshold: float,
    seed: int,
    regime_A_scales: np.ndarray = None,
    policy_lag: int = 3,
    policy_decay: float = 0.7,
    stress_growth_coupling: float = 0.15,
    student_t_df: float = 5.0,
) -> list:
    """Pure numpy fallback with all structural fixes applied."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    if regime_A_scales is None:
        regime_A_scales = np.array([1.0, 0.97, 0.92])

    # Build delayed policy impulse array
    policy_impulse = np.zeros(H)
    for t in range(H):
        lag_t = t - policy_lag
        if lag_t >= 0:
            policy_impulse[t] = delta_bps * (policy_decay ** lag_t)

    # Initialize
    Z0 = rng.standard_normal((N, n))
    X = mu0[np.newaxis, :] + Z0 @ P0_chol.T

    regimes = np.full(N, initial_regime, dtype=int)
    n_spag = min(30, N)

    results_per_step = []

    for step in range(H):
        # 1) Regime transitions FIRST
        if regime_switching:
            u_rand = rng.random(N)
            new_regimes = np.zeros(N, dtype=int)
            for i in range(N):
                curr = regimes[i]
                for j in range(3):
                    if u_rand[i] <= Pi_cumsum[curr, j]:
                        new_regimes[i] = j
                        break
                else:
                    new_regimes[i] = 2
            regimes = new_regimes

        # 2) Student-t noise (fat tails)
        Z_norm = rng.standard_normal((N, n))
        chi2_samples = rng.chisquare(df=student_t_df, size=N)
        chi2_samples = np.maximum(chi2_samples, 0.01)
        t_scales = np.sqrt(student_t_df / chi2_samples)  # (N,)

        if regime_switching:
            scales = noise_scales[regimes]
            noise = (Z_norm @ Q_chol.T) * t_scales[:, np.newaxis] * np.sqrt(scales[:, np.newaxis])
        else:
            noise = (Z_norm @ Q_chol.T) * t_scales[:, np.newaxis]

        # 3) Save current growth for indirect stress coupling
        growth_current = X[:, 2].copy() if n > 2 else np.zeros(N)

        # 4) Mean-reverting transition with regime-dependent A scaling
        #    NO constant Bu offset
        if regime_switching:
            X_new = np.zeros_like(X)
            for r in range(3):
                mask = (regimes == r)
                if np.any(mask):
                    X_new[mask] = X[mask] @ (A * regime_A_scales[r]).T
            X = X_new + noise
        else:
            X = X @ A.T + noise

        # 5) Delayed policy effect on GROWTH channel only
        if n > 2:
            X[:, 2] += B[2, 0] * policy_impulse[step]

        # 6) Indirect stress from growth
        if n > 2:
            X[:, 0] += stress_growth_coupling * (-growth_current)

        # Statistics
        stress = X[:, 0]
        growth = X[:, 2] if n > 2 else np.zeros(N)

        stress_fan = {
            "p5": float(np.percentile(stress, 5)),
            "p25": float(np.percentile(stress, 25)),
            "p50": float(np.percentile(stress, 50)),
            "p75": float(np.percentile(stress, 75)),
            "p95": float(np.percentile(stress, 95)),
        }
        growth_fan = {
            "p5": float(np.percentile(growth, 5)),
            "p25": float(np.percentile(growth, 25)),
            "p50": float(np.percentile(growth, 50)),
            "p75": float(np.percentile(growth, 75)),
            "p95": float(np.percentile(growth, 95)),
        }

        # Crisis prob: stress > threshold OR regime == Crisis
        crisis_mask = (stress > crisis_threshold)
        if regime_switching:
            crisis_mask = crisis_mask | (regimes == 2)
        cp = float(np.mean(crisis_mask))

        # Empirical ES95: mean of stress where stress >= VaR95
        var95 = float(np.percentile(stress, 95))
        tail_mask = stress >= var95
        es95 = float(np.mean(stress[tail_mask])) if np.any(tail_mask) else var95

        spaghetti_step = [
            {"id": int(i), "stress": float(X[i, 0])} for i in range(n_spag)
        ]

        results_per_step.append({
            "step": step + 1,
            "stress_fan": stress_fan,
            "growth_fan": growth_fan,
            "crisis_prob": cp,
            "es95_stress": es95,
            "spaghetti": spaghetti_step,
        })

    return results_per_step


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for policy evaluation.

    Supports both Numba-accelerated and pure numpy backends.
    Provides step-by-step results for WebSocket streaming and
    batch results for policy recommendation.
    """

    def __init__(
        self,
        A_daily: np.ndarray,
        B_daily: np.ndarray,
        Q_daily: np.ndarray,
        mu_T: np.ndarray,
        P_T: np.ndarray,
        crisis_threshold: float,
        stress_std: float = 1.0,
    ):
        self.n = A_daily.shape[0]
        self.stress_std = stress_std

        # Crisis threshold: use a softer multiplier (1.0 × std) so that
        # the MC-predicted crisis_prob better aligns with the ~15% base rate
        # of the realized crisis indicator (expanding 85th-percentile stress).
        # Previously 1.5× produced probabilities too low relative to realized.
        self.crisis_threshold = 1.0 * stress_std
        logger.info(
            f"[CrisisThreshold] Using 1.0 × stress_std = 1.0 × {stress_std:.4f} "
            f"= {self.crisis_threshold:.4f} (was {crisis_threshold:.4f})"
        )

        # Convert to monthly (stationarity enforced inside)
        self.A_monthly, self.Q_monthly, self.B_monthly = daily_to_monthly_params(
            A_daily, Q_daily, B_daily, settings.trading_days_per_month
        )

        self.mu_T = np.array(mu_T)
        self.P_T = np.array(P_T)

        # Cholesky decompositions for sampling
        self.Q_chol = np.linalg.cholesky(
            self.Q_monthly + 1e-8 * np.eye(self.n)
        )
        self.P0_chol = np.linalg.cholesky(
            self.P_T + 1e-8 * np.eye(self.n)
        )

        self.regime_model = RegimeModel()

        # New config values
        self.regime_A_scales = np.array(settings.regime_A_scales, dtype=np.float64)
        self.policy_lag = settings.policy_lag_months
        self.policy_decay = settings.policy_decay
        self.stress_growth_coupling = settings.stress_growth_coupling
        self.student_t_df = settings.student_t_df

    def apply_shocks(
        self,
        mu0: np.ndarray,
        shocks: dict,
    ) -> tuple:
        """
        Apply shock injections to initial state.

        Args:
            mu0: initial state mean
            shocks: dict with keys "credit", "vol", "rate" and intensity values (in sigma units)

        Returns:
            (modified_mu0, additional_bps)
        """
        mu = mu0.copy()
        extra_bps = 0.0

        sigma = self.stress_std

        if "credit" in shocks and shocks["credit"] != 0:
            k = shocks["credit"]
            mu[0] += settings.shock_credit_stress * k * sigma
            mu[1] += settings.shock_credit_liquidity * k * sigma

        if "vol" in shocks and shocks["vol"] != 0:
            k = shocks["vol"]
            mu[0] += settings.shock_vol_stress * k * sigma

        if "rate" in shocks and shocks["rate"] != 0:
            k = shocks["rate"]
            extra_bps += settings.shock_rate_bps * k

        return mu, extra_bps

    def simulate_streaming(
        self,
        delta_bps: float = 0.0,
        N: int = 5000,
        H: int = 24,
        shocks: dict = None,
        regime_switching: bool = True,
        seed: int = 42,
    ) -> list[dict]:
        """
        Run full simulation and return per-step results for streaming.

        Returns list of dicts, one per timestep.
        """
        mu0 = self.mu_T.copy()
        extra_bps = 0.0

        if shocks:
            mu0, extra_bps = self.apply_shocks(mu0, shocks)

        total_bps = delta_bps + extra_bps
        initial_regime = self.regime_model.initial_regime(mu0[0])

        Pi_cumsum = self.regime_model.Pi_cumsum
        noise_scales = self.regime_model.scales

        # Try Numba path first
        if NUMBA_AVAILABLE:
            try:
                result = _simulate_paths_numba(
                    self.A_monthly, self.B_monthly, self.Q_chol,
                    mu0, self.P0_chol, total_bps,
                    N, H, regime_switching,
                    Pi_cumsum, noise_scales, initial_regime,
                    self.crisis_threshold, seed,
                    self.regime_A_scales,
                    self.policy_lag,
                    self.policy_decay,
                    self.stress_growth_coupling,
                    self.student_t_df,
                )

                (sp5, sp25, sp50, sp75, sp95,
                 gp5, gp25, gp50, gp75, gp95,
                 crisis_prob, es95, spaghetti) = result

                steps = []
                n_spag = min(30, N)
                for step in range(H):
                    spag = [
                        {"id": int(i), "stress": float(spaghetti[step, i])}
                        for i in range(n_spag)
                    ]
                    steps.append({
                        "step": step + 1,
                        "H": H,
                        "stress_fan": {
                            "p5": float(sp5[step]),
                            "p25": float(sp25[step]),
                            "p50": float(sp50[step]),
                            "p75": float(sp75[step]),
                            "p95": float(sp95[step]),
                        },
                        "growth_fan": {
                            "p5": float(gp5[step]),
                            "p25": float(gp25[step]),
                            "p50": float(gp50[step]),
                            "p75": float(gp75[step]),
                            "p95": float(gp95[step]),
                        },
                        "crisis_prob": float(crisis_prob[step]),
                        "es95_stress": float(es95[step]),
                        "spaghetti": spag,
                    })
                return steps
            except Exception as e:
                logger.warning(f"Numba simulation failed, falling back to numpy: {e}")

        # Numpy fallback
        results = simulate_paths_numpy(
            self.A_monthly, self.B_monthly, self.Q_chol,
            mu0, self.P0_chol, total_bps,
            N, H, regime_switching,
            Pi_cumsum, noise_scales, initial_regime,
            self.crisis_threshold, seed,
            regime_A_scales=self.regime_A_scales,
            policy_lag=self.policy_lag,
            policy_decay=self.policy_decay,
            stress_growth_coupling=self.stress_growth_coupling,
            student_t_df=self.student_t_df,
        )

        for r in results:
            r["H"] = H

        return results

    def evaluate_policy(
        self,
        delta_bps: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        lam: float = 1.0,
        N: int = 5000,
        H: int = 24,
        shocks: dict = None,
        regime_switching: bool = True,
        seed: int = 42,
    ) -> dict:
        """
        Evaluate a single policy action and compute loss components.

        Loss = alpha * mean_stress + beta * growth_penalty + gamma * ES95 + lambda * crisis_end

        Returns dict with loss breakdown.
        """
        steps = self.simulate_streaming(
            delta_bps=delta_bps, N=N, H=H, shocks=shocks,
            regime_switching=regime_switching, seed=seed,
        )

        # Aggregate metrics across steps
        mean_stress = np.mean([s["stress_fan"]["p50"] for s in steps])
        # Growth penalty: penalize negative growth
        mean_growth_penalty = np.mean([max(0, -s["growth_fan"]["p50"]) for s in steps])
        mean_es95 = np.mean([s["es95_stress"] for s in steps])
        crisis_end = steps[-1]["crisis_prob"]

        total_loss = (
            alpha * mean_stress
            + beta * mean_growth_penalty
            + gamma * mean_es95
            + lam * crisis_end
        )

        return {
            "delta_bps": delta_bps,
            "mean_stress": float(mean_stress),
            "mean_growth_penalty": float(mean_growth_penalty),
            "mean_es95": float(mean_es95),
            "crisis_end": float(crisis_end),
            "total_loss": float(total_loss),
            "crisis_prob_path": [float(s["crisis_prob"]) for s in steps],
        }
