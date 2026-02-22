# PolEn Backtest Report

**Generated**: 2026-02-21 14:29
**Period**: 2002-08-31 → 2025-02-28
**Forecasts**: 91
**Rolling window**: 120 months  |  **Horizon**: 12 months  |  **MC paths**: 500

## 1. Crisis Probability Forecasts

| Metric | Value | Interpretation |
|--------|------:|----------------|
| Brier Score | 0.2555 | 0 = perfect, 0.25 = random |
| AUC | 0.5295 | >0.5 better than coin flip |
| Log Score | 1.0630 | Lower is better |

**Calibration (predicted vs realized):**

| Bin | Pred Mean | Emp Freq | Count |
|-----|----------:|---------:|------:|
| [0.0, 0.1) | 0.015 | 0.250 | 8 |
| [0.1, 0.2) | 0.153 | 0.500 | 6 |
| [0.2, 0.3) | 0.250 | 0.321 | 28 |
| [0.3, 0.4) | 0.339 | 0.429 | 7 |
| [0.4, 0.5) | 0.437 | 0.222 | 27 |
| [0.5, 0.6) | 0.536 | 0.500 | 10 |
| [0.6, 0.7) | 0.636 | 1.000 | 1 |
| [0.7, 0.8) | 0.739 | 0.000 | 3 |
| [0.8, 0.9) | 0.850 | — | 0 |
| [0.9, 1.0) | 0.904 | 0.000 | 1 |

## 2. Expected Shortfall (ES95) Validation

| Metric | Value | Target / Interpretation |
|--------|------:|------------------------|
| Hit Ratio | 0.0330 | Target: 0.05 |
| Kupiec p-value | 0.4279 | ✓ PASS (>0.05 = correct coverage) |
| Christoffersen p-value | 0.6492 | ✓ PASS (>0.05 = independent violations) |
| Mean Shortfall Error | -4.2991 | Negative = conservative |

## 3. Regime Classification vs NBER Recessions

| Metric | Value |
|--------|------:|
| Precision | 0.107 |
| Recall | 0.667 |
| F1 Score | 0.185 |
| AUC | 0.603 |

**Confusion Matrix** (rows = predicted, cols = actual):

|  | Actual Normal | Actual Recession |
|--|-------------:|-----------------:|
| Pred Normal | 32 | 50 |
| Pred Crisis | 3 | 6 |

## 4. Policy Comparison (Active vs Passive Hold)

| Metric | Active | Passive | Better? |
|--------|-------:|--------:|---------|
| Avg Loss | 5.1031 | 5.3643 | ✓ Active |
| Max Drawdown | 7.3960 | 7.4484 | ✓ Active |
| ES95 (tail risk) | 6.7872 | 6.9666 | ✓ Active |

**Recommended action distribution:**

- -150 bps: 37 times (40.7%)
- +150 bps: 54 times (59.3%)

## 5. Historical Stress Episode Analysis

### GFC_2008 (2007-07 to 2009-06)
- Model max crisis probability: **0.352**
- Model max ES95: **6.047**
- Realized stress (max): 3.493
- Realized stress (mean): 0.211
- Months in episode (realized): 23

### COVID_2020 (2020-01 to 2020-09)
- Model max crisis probability: **0.350**
- Model max ES95: **5.495**
- Realized stress (max): 6.299
- Realized stress (mean): 0.968
- Months in episode (realized): 8

### Tightening_2022 (2022-01 to 2022-12)
- Model max crisis probability: **0.484**
- Model max ES95: **5.905**
- Realized stress (max): 2.982
- Realized stress (mean): -0.273
- Months in episode (realized): 11

## 6. Success Criteria Summary

**Overall: 4/4 criteria passed**

| Criterion | Value | Threshold | Result |
|-----------|------:|----------:|--------|
| crisis_auc_gt_baseline | 0.5295 | 0.5000 | ✓ PASS |
| es95_kupiec_passes | 0.4279 | 0.0500 | ✓ PASS |
| regime_f1_nontrivial | 0.1846 | 0.0000 | ✓ PASS |
| policy_reduces_loss | 5.1031 | 5.3643 | ✓ PASS |
