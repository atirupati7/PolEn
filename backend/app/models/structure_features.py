"""
Cross-asset structure feature extraction.

Computes rolling covariance/correlation matrices, eigenvalue decomposition,
and structural metrics (correlation compression, concentration ratio, etc.)
from the market vector Y_t.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import linalg

from app.config import settings

logger = logging.getLogger(__name__)

# Market vector columns used for covariance/correlation analysis
# Y_t = [r_spx, d_DGS2, d_DGS10, cs, dvix, r_usd]
MARKET_VECTOR_COLS = ["z_r_spx", "z_d_DGS2", "z_d_DGS10", "z_cs", "z_dvix", "z_r_usd"]
MARKET_VECTOR_LABELS = ["SPX", "2Y", "10Y", "Credit", "VIX", "USD"]


def compute_structure_features(
    df: pd.DataFrame,
    window: int = 60,
    shrinkage_delta: float = 0.01,
) -> dict:
    """
    Compute cross-asset structure features Z_t from the processed dataframe.

    At each day t (we compute for the latest available window), constructs:
    - Rolling covariance matrix Σ_t
    - Rolling correlation matrix R_t
    - Eigenvalue decomposition of R_t
    - Structural metrics

    Args:
        df: Processed DataFrame with z-score columns
        window: Rolling window size (default 60 business days)
        shrinkage_delta: Ledoit-Wolf-style shrinkage parameter

    Returns:
        Dictionary with:
        - Z_t: structural feature vector (numpy array)
        - R_t: latest correlation matrix
        - cov_t: latest covariance matrix
        - eigenvalues: sorted eigenvalues of R_t
        - labels: column labels for the market vector
        - metrics: dict of named structural metrics
        - Z_history: DataFrame of Z_t over time
    """
    # Select available market vector columns
    available_cols = [c for c in MARKET_VECTOR_COLS if c in df.columns]
    available_labels = [
        MARKET_VECTOR_LABELS[i]
        for i, c in enumerate(MARKET_VECTOR_COLS)
        if c in df.columns
    ]

    if len(available_cols) < 3:
        raise ValueError(
            f"Need at least 3 market vector columns, got {len(available_cols)}: {available_cols}"
        )

    market_data = df[available_cols].dropna()
    n = len(market_data)

    if n < window:
        raise ValueError(
            f"Not enough data ({n} rows) for rolling window ({window})"
        )

    # Compute rolling features for the FULL available history (not just the
    # last ~2 years).  This is critical for backtesting over decades.
    # Sub-sampling: compute every `step` days and then resample to monthly for
    # Z_history to keep memory and runtime manageable for long histories.
    step = max(1, 5 if n > 5000 else 1)   # every 5 bdays ≈ weekly for long histories
    data_slice = market_data

    feature_records = []
    dates = []
    corr_records = []
    eigen_records = []

    d = len(available_cols)

    for i in range(window, len(data_slice), step):
        w = data_slice.iloc[i - window : i].values  # shape (window, d)

        # Covariance matrix
        cov_mat = np.cov(w, rowvar=False)

        # Apply shrinkage for numerical stability
        trace_cov = np.trace(cov_mat)
        target = np.eye(d) * trace_cov / d
        cov_mat = (1 - shrinkage_delta) * cov_mat + shrinkage_delta * target

        # Correlation matrix from (possibly shrunk) covariance
        std_diag = np.sqrt(np.diag(cov_mat))
        std_diag[std_diag < 1e-10] = 1e-10
        corr_mat = cov_mat / np.outer(std_diag, std_diag)
        np.fill_diagonal(corr_mat, 1.0)
        # Clip to valid range
        corr_mat = np.clip(corr_mat, -1.0, 1.0)

        # Eigenvalue decomposition
        try:
            eigenvalues = np.sort(np.real(linalg.eigvalsh(corr_mat)))[::-1]
        except linalg.LinAlgError:
            eigenvalues = np.ones(d) / d

        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

        # Structural metrics
        lambda_sum = eigenvalues.sum()
        lambda_concentration = eigenvalues[0] / lambda_sum if lambda_sum > 0 else 1.0
        lambda_dispersion = np.std(eigenvalues) / (np.mean(eigenvalues) + 1e-10)

        # Off-diagonal correlation statistics
        mask = ~np.eye(d, dtype=bool)
        offdiag = corr_mat[mask]
        corr_mean_offdiag = np.mean(offdiag)
        corr_std_offdiag = np.std(offdiag)

        # Additional features from the data
        row = data_slice.iloc[i]
        cs_level = row.get("z_cs", 0.0) if "z_cs" in available_cols else 0.0
        # slope from the main df at the matching date
        date_i = data_slice.index[i]
        if "slope" in df.columns and date_i in df.index:
            slope_val = float(df.loc[date_i, "slope"])
        else:
            slope_val = 0.0

        # Rolling std of dvix (volatility of volatility)
        if "z_dvix" in available_cols:
            dvix_idx = available_cols.index("z_dvix")
            vix_vol = np.std(w[:, dvix_idx])
        else:
            vix_vol = 0.0

        feature_vec = np.array([
            lambda_concentration,
            lambda_dispersion,
            corr_mean_offdiag,
            corr_std_offdiag,
            cs_level,
            slope_val,
            vix_vol,
            eigenvalues[0],
        ])

        feature_records.append(feature_vec)
        dates.append(data_slice.index[i])
        corr_records.append(corr_mat.tolist())
        eigen_records.append(eigenvalues.tolist())

    feature_names = [
        "lambda_concentration",
        "lambda_dispersion",
        "corr_mean_offdiag",
        "corr_std_offdiag",
        "cs_level",
        "slope",
        "vix_vol",
        "lambda1",
    ]

    Z_history = pd.DataFrame(feature_records, index=dates, columns=feature_names)

    # Resample to end-of-month so downstream (Kalman, backtest) sees monthly data
    Z_history = Z_history.resample("M").last().dropna(how="all")

    # Build monthly correlation / eigenvalue snapshots.
    # Later entries in the same month overwrite earlier ones (i.e. we keep
    # the last observation of each month, matching Z_history's resample).
    monthly_corr_snapshots: dict = {}
    for idx_snap, snap_date in enumerate(dates):
        month_key = snap_date.strftime("%Y-%m")
        monthly_corr_snapshots[month_key] = {
            "date": snap_date.strftime("%Y-%m-%d"),
            "correlation_matrix": corr_records[idx_snap],
            "eigenvalues": eigen_records[idx_snap],
        }

    # Latest values
    latest_Z = feature_records[-1]

    # Recompute latest correlation/covariance for API
    w_latest = data_slice.iloc[-window:].values
    cov_latest = np.cov(w_latest, rowvar=False)
    trace_cov = np.trace(cov_latest)
    target = np.eye(d) * trace_cov / d
    cov_latest = (1 - shrinkage_delta) * cov_latest + shrinkage_delta * target

    std_diag = np.sqrt(np.diag(cov_latest))
    std_diag[std_diag < 1e-10] = 1e-10
    corr_latest = cov_latest / np.outer(std_diag, std_diag)
    np.fill_diagonal(corr_latest, 1.0)
    corr_latest = np.clip(corr_latest, -1.0, 1.0)

    eigenvalues_latest = np.sort(np.real(linalg.eigvalsh(corr_latest)))[::-1]
    eigenvalues_latest = np.maximum(eigenvalues_latest, 0)

    metrics = {
        "lambda_concentration": float(latest_Z[0]),
        "lambda_dispersion": float(latest_Z[1]),
        "corr_mean_offdiag": float(latest_Z[2]),
        "corr_std_offdiag": float(latest_Z[3]),
        "cs_level": float(latest_Z[4]),
        "slope": float(latest_Z[5]),
        "vix_vol": float(latest_Z[6]),
        "lambda1": float(latest_Z[7]),
    }

    return {
        "Z_t": latest_Z,
        "R_t": corr_latest,
        "cov_t": cov_latest,
        "eigenvalues": eigenvalues_latest.tolist(),
        "labels": available_labels,
        "metrics": metrics,
        "Z_history": Z_history,
        "feature_names": feature_names,
        "monthly_corr_snapshots": monthly_corr_snapshots,
    }
