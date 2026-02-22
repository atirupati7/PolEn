"""Tests for cross-asset structure feature extraction."""

import numpy as np
import pandas as pd
import pytest

from app.models.structure_features import compute_structure_features


def _make_test_dataframe(n: int = 500) -> pd.DataFrame:
    """Generate a test dataframe with z-scored market data."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2025-12-31", periods=n, freq="B")

    # Create correlated features
    factor = np.random.randn(n)
    data = {
        "z_r_spx": 0.6 * factor + 0.4 * np.random.randn(n),
        "z_d_DGS2": 0.3 * factor + 0.7 * np.random.randn(n),
        "z_d_DGS10": 0.4 * factor + 0.6 * np.random.randn(n),
        "z_cs": -0.5 * factor + 0.5 * np.random.randn(n),
        "z_dvix": -0.4 * factor + 0.6 * np.random.randn(n),
        "z_r_usd": 0.2 * factor + 0.8 * np.random.randn(n),
        "slope": 1.0 + 0.3 * np.cumsum(np.random.randn(n) * 0.01),
    }
    return pd.DataFrame(data, index=dates)


class TestStructureFeatures:
    """Tests for correlation and eigenvalue computations."""

    def test_correlation_matrix_symmetric(self):
        """Correlation matrix R_t must be symmetric."""
        df = _make_test_dataframe()
        result = compute_structure_features(df, window=60)
        R = result["R_t"]
        assert np.allclose(R, R.T, atol=1e-10), "Correlation matrix not symmetric"

    def test_correlation_values_valid(self):
        """All correlation values must be in [-1, 1]."""
        df = _make_test_dataframe()
        result = compute_structure_features(df, window=60)
        R = result["R_t"]
        assert np.all(R >= -1.0 - 1e-10), f"Min corr={R.min()}"
        assert np.all(R <= 1.0 + 1e-10), f"Max corr={R.max()}"

    def test_correlation_diagonal_ones(self):
        """Diagonal of correlation matrix must be 1."""
        df = _make_test_dataframe()
        result = compute_structure_features(df, window=60)
        R = result["R_t"]
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-6)

    def test_eigenvalues_non_negative(self):
        """Eigenvalues of correlation matrix must be non-negative (within tolerance)."""
        df = _make_test_dataframe()
        result = compute_structure_features(df, window=60)
        eigenvalues = np.array(result["eigenvalues"])
        assert np.all(eigenvalues >= -1e-8), f"Negative eigenvalue: {eigenvalues.min()}"

    def test_eigenvalues_sum_to_dimension(self):
        """Eigenvalues of correlation matrix should sum to the dimension."""
        df = _make_test_dataframe()
        result = compute_structure_features(df, window=60)
        eigenvalues = np.array(result["eigenvalues"])
        n_assets = len(result["labels"])
        np.testing.assert_allclose(eigenvalues.sum(), n_assets, atol=0.5)

    def test_feature_vector_shape(self):
        """Z_t feature vector should have correct dimension."""
        df = _make_test_dataframe()
        result = compute_structure_features(df, window=60)
        assert len(result["Z_t"]) == len(result["feature_names"])

    def test_z_history_has_correct_columns(self):
        """Z_history DataFrame should have named feature columns."""
        df = _make_test_dataframe()
        result = compute_structure_features(df, window=60)
        Z_hist = result["Z_history"]
        assert isinstance(Z_hist, pd.DataFrame)
        assert len(Z_hist) > 0
        assert list(Z_hist.columns) == result["feature_names"]

    def test_metrics_dict_complete(self):
        """Metrics dict should contain all expected keys."""
        df = _make_test_dataframe()
        result = compute_structure_features(df, window=60)
        expected_keys = {
            "lambda_concentration", "lambda_dispersion",
            "corr_mean_offdiag", "corr_std_offdiag",
            "cs_level", "slope", "vix_vol", "lambda1",
        }
        assert set(result["metrics"].keys()) == expected_keys

    def test_lambda_concentration_valid(self):
        """Lambda concentration must be between 0 and 1."""
        df = _make_test_dataframe()
        result = compute_structure_features(df, window=60)
        lc = result["metrics"]["lambda_concentration"]
        assert 0 < lc <= 1.0, f"lambda_concentration={lc}"

    def test_shrinkage_handles_degenerate_data(self):
        """Shrinkage should prevent issues with near-singular covariance."""
        df = _make_test_dataframe(200)
        # Make two columns nearly identical
        df["z_d_DGS2"] = df["z_d_DGS10"] + 1e-8 * np.random.randn(len(df))
        result = compute_structure_features(df, window=60, shrinkage_delta=0.05)
        R = result["R_t"]
        assert np.all(np.isfinite(R)), "Non-finite values in correlation matrix"
