"""Tests for Monte Carlo simulation engine and policy evaluation."""

import numpy as np
import pytest

from app.models.monte_carlo import MonteCarloEngine, daily_to_monthly_params


def _make_test_engine(
    stress_level: float = 0.0,
    noise_scale: float = 0.1,
) -> MonteCarloEngine:
    """Create a test MonteCarloEngine with known parameters."""
    n = 3
    A_daily = 0.98 * np.eye(n)
    B_daily = np.array([[0.003], [0.006], [-0.004]])
    Q_daily = np.eye(n) * noise_scale

    mu_T = np.array([stress_level, 0.0, 0.5])
    P_T = np.eye(n) * 0.05

    return MonteCarloEngine(
        A_daily=A_daily,
        B_daily=B_daily,
        Q_daily=Q_daily,
        mu_T=mu_T,
        P_T=P_T,
        crisis_threshold=2.0,
        stress_std=1.0,
    )


class TestDailyToMonthly:
    """Tests for parameter conversion."""

    def test_A_monthly_shape(self):
        A = 0.98 * np.eye(3)
        Q = np.eye(3) * 0.1
        B = np.array([[0.003], [0.006], [-0.004]])

        A_m, Q_m, B_m = daily_to_monthly_params(A, Q, B)

        assert A_m.shape == (3, 3)
        assert Q_m.shape == (3, 3)
        assert B_m.shape == (3, 1)

    def test_Q_monthly_symmetric(self):
        A = 0.98 * np.eye(3)
        Q = np.eye(3) * 0.1
        B = np.array([[0.003], [0.006], [-0.004]])

        _, Q_m, _ = daily_to_monthly_params(A, Q, B)

        np.testing.assert_allclose(Q_m, Q_m.T, atol=1e-10)

    def test_Q_monthly_positive_definite(self):
        A = 0.98 * np.eye(3)
        Q = np.eye(3) * 0.1
        B = np.array([[0.003], [0.006], [-0.004]])

        _, Q_m, _ = daily_to_monthly_params(A, Q, B)

        eigenvalues = np.linalg.eigvalsh(Q_m)
        assert np.all(eigenvalues > 0), f"Q_monthly not PD: {eigenvalues}"


class TestMonteCarloEngine:
    """Tests for the Monte Carlo simulation engine."""

    def test_simulate_returns_correct_steps(self):
        """Simulation should return H steps."""
        engine = _make_test_engine()
        steps = engine.simulate_streaming(delta_bps=0, N=500, H=12, regime_switching=False)
        assert len(steps) == 12

    def test_step_has_required_fields(self):
        """Each step should contain all required fields."""
        engine = _make_test_engine()
        steps = engine.simulate_streaming(delta_bps=0, N=500, H=6, regime_switching=False)
        required = {"step", "H", "stress_fan", "growth_fan", "crisis_prob", "es95_stress", "spaghetti"}
        for s in steps:
            assert required.issubset(set(s.keys())), f"Missing keys: {required - set(s.keys())}"

    def test_fan_has_percentiles(self):
        """Fan chart data should have p5, p25, p50, p75, p95."""
        engine = _make_test_engine()
        steps = engine.simulate_streaming(delta_bps=0, N=500, H=6, regime_switching=False)
        fan_keys = {"p5", "p25", "p50", "p75", "p95"}
        for s in steps:
            assert set(s["stress_fan"].keys()) == fan_keys
            assert set(s["growth_fan"].keys()) == fan_keys

    def test_percentile_ordering(self):
        """Percentiles must be ordered: p5 <= p25 <= p50 <= p75 <= p95."""
        engine = _make_test_engine()
        steps = engine.simulate_streaming(delta_bps=0, N=1000, H=12, regime_switching=False)
        for s in steps:
            fan = s["stress_fan"]
            assert fan["p5"] <= fan["p25"] <= fan["p50"] <= fan["p75"] <= fan["p95"], (
                f"Bad ordering at step {s['step']}: {fan}"
            )

    def test_crisis_prob_increases_with_stress_shock(self):
        """Higher initial stress should increase crisis probability."""
        engine_low = _make_test_engine(stress_level=0.0)
        engine_high = _make_test_engine(stress_level=2.5)

        steps_low = engine_low.simulate_streaming(delta_bps=0, N=2000, H=12, regime_switching=False, seed=42)
        steps_high = engine_high.simulate_streaming(delta_bps=0, N=2000, H=12, regime_switching=False, seed=42)

        avg_crisis_low = np.mean([s["crisis_prob"] for s in steps_low])
        avg_crisis_high = np.mean([s["crisis_prob"] for s in steps_high])

        assert avg_crisis_high > avg_crisis_low, (
            f"High stress ({avg_crisis_high}) should have higher crisis prob than low ({avg_crisis_low})"
        )

    def test_spaghetti_paths_count(self):
        """Should return up to 30 spaghetti paths."""
        engine = _make_test_engine()
        steps = engine.simulate_streaming(delta_bps=0, N=500, H=6, regime_switching=False)
        for s in steps:
            assert len(s["spaghetti"]) == 30

    def test_crisis_prob_in_valid_range(self):
        """Crisis probability must be in [0, 1]."""
        engine = _make_test_engine()
        steps = engine.simulate_streaming(delta_bps=0, N=500, H=12, regime_switching=True)
        for s in steps:
            assert 0 <= s["crisis_prob"] <= 1, f"Invalid crisis_prob: {s['crisis_prob']}"

    def test_regime_switching_increases_variance(self):
        """Regime switching should generally increase the spread of the fan chart."""
        engine = _make_test_engine()

        steps_no_rs = engine.simulate_streaming(
            delta_bps=0, N=2000, H=12, regime_switching=False, seed=42
        )
        steps_rs = engine.simulate_streaming(
            delta_bps=0, N=2000, H=12, regime_switching=True, seed=42
        )

        spread_no_rs = np.mean([
            s["stress_fan"]["p95"] - s["stress_fan"]["p5"] for s in steps_no_rs
        ])
        spread_rs = np.mean([
            s["stress_fan"]["p95"] - s["stress_fan"]["p5"] for s in steps_rs
        ])

        # Regime switching should not decrease spread significantly
        assert spread_rs >= spread_no_rs * 0.8, (
            f"RS spread ({spread_rs}) much smaller than no-RS ({spread_no_rs})"
        )


class TestPolicyEvaluation:
    """Tests for policy loss computation."""

    def test_evaluate_policy_returns_loss(self):
        """Policy evaluation must return total_loss."""
        engine = _make_test_engine()
        result = engine.evaluate_policy(delta_bps=0, N=500, H=12)
        assert "total_loss" in result
        assert "mean_stress" in result
        assert "crisis_end" in result
        assert np.isfinite(result["total_loss"])

    def test_policy_loss_consistent_with_components(self):
        """Total loss should equal weighted sum of components (with default weights)."""
        engine = _make_test_engine()
        result = engine.evaluate_policy(
            delta_bps=0, alpha=1.0, beta=1.0, gamma=1.0, lam=1.0, N=500, H=12
        )

        expected = (
            result["mean_stress"]
            + result["mean_growth_penalty"]
            + result["mean_es95"]
            + result["crisis_end"]
        )

        np.testing.assert_allclose(result["total_loss"], expected, atol=1e-6)

    def test_easing_reduces_stress_vs_tightening(self):
        """Easing policy should generally produce lower stress than tightening."""
        engine = _make_test_engine(stress_level=1.0)

        result_ease = engine.evaluate_policy(delta_bps=-50, N=2000, H=12, seed=42)
        result_tight = engine.evaluate_policy(delta_bps=50, N=2000, H=12, seed=42)

        # Easing should produce lower (or at least not much higher) mean stress
        # Given B = [0.003, 0.006, -0.004], positive bps increases stress
        assert result_ease["mean_stress"] <= result_tight["mean_stress"] + 0.5, (
            f"Ease stress ({result_ease['mean_stress']}) should be <= Tighten ({result_tight['mean_stress']})"
        )
