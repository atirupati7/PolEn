"""
Generate CSV data files for Figma Make visualization.
Pulls real data from the live PolEn backend API.
Outputs to backend/scripts/figma_data/
"""

import csv
import json
import math
import os
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "http://localhost:8000"
OUT_DIR = Path(__file__).parent / "figma_data"
OUT_DIR.mkdir(exist_ok=True)


def fetch_json(path, method="GET", body=None):
    url = BASE_URL + path
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    if body:
        req.data = json.dumps(body).encode()
    with urllib.request.urlopen(req) as r:
        return json.load(r)


# ── 1. Fetch raw data ────────────────────────────────────────────────────────

print("Fetching current macro state...")
state = fetch_json("/api/state/current")

print("Fetching 10-year timeseries...")
ts = fetch_json("/api/historical/timeseries?years=10")

print("Fetching policy evaluation (Normal regime)...")
policy_eval = fetch_json("/api/policy/recommend", method="POST", body={"policy": "neutral"})


# ── 2. Build macro_history.csv ───────────────────────────────────────────────

def estimate_regime(stress_score):
    """Map a 0-1 stress score to a regime label."""
    if stress_score < 0.35:
        return "Normal"
    elif stress_score < 0.65:
        return "Fragile"
    else:
        return "Crisis"


def normalize(values):
    """Min-max normalize a list to [0, 1]."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


dates = ts["dates"]
s = ts["series"]

infl   = s["inflation_yoy"]["values"]
fed    = s["fed_rate"]["values"]
vix    = s["vix_level"]["values"]
slope  = s["slope"]["values"]
rspx   = s["r_spx"]["values"]
cs     = s["cs"]["values"]
dgs2   = s["DGS2"]["values"]
dgs10  = s["DGS10"]["values"]

# Compute a rolling stress proxy: blend normalised VIX + credit spread
vix_n  = normalize(vix)
cs_n   = normalize(cs)
slope_n = normalize(slope)  # inverted: low slope → more stress
stress_proxy = [0.5 * vn + 0.3 * cn + 0.2 * (1 - sn)
                for vn, cn, sn in zip(vix_n, cs_n, slope_n)]

rows_history = []
for i, d in enumerate(dates):
    rows_history.append({
        "date": d,
        "inflation_yoy": round(infl[i] * 100, 2),      # pct
        "fed_rate_pct": round(fed[i] * 100, 2),         # pct
        "vix": round(vix[i], 2),
        "yield_slope_2_10": round(slope[i], 3),
        "credit_spread": round(cs[i], 3),
        "spx_monthly_ret_pct": round(rspx[i] * 100, 3),
        "stress_score": round(stress_proxy[i], 4),
        "regime": estimate_regime(stress_proxy[i]),
    })

history_path = OUT_DIR / "macro_history.csv"
with open(history_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows_history[0].keys())
    writer.writeheader()
    writer.writerows(rows_history)
print(f"  Written {len(rows_history)} rows → {history_path}")


# ── 3. Build policy_fanChart.csv ─────────────────────────────────────────────
# Fan chart: Ease / Hold / Tighten × 24-month stress trajectories
# Uses real policy-comparison mean_stress as the 12-month anchor,
# then linearly interpolates to 24 months using regime decay rates.

comparison = {row["action"]: row for row in policy_eval["comparison"]}

# Decay constants derived from RL dynamics (Fragile noise scale 1.8×, monthly)
REGIME_PARAMS = {
    "Ease":    {"initial_stress": state["stress_score"], "monthly_delta": comparison["Ease"]["mean_stress"] / 12,    "crisis_prob_12m": comparison["Ease"]["crisis_end"]},
    "Hold":    {"initial_stress": state["stress_score"], "monthly_delta": comparison["Hold"]["mean_stress"] / 12,    "crisis_prob_12m": comparison["Hold"]["crisis_end"]},
    "Tighten": {"initial_stress": state["stress_score"], "monthly_delta": comparison["Tighten"]["mean_stress"] / 12, "crisis_prob_12m": comparison["Tighten"]["crisis_end"]},
}

# After 12m the delta halves (mean-reversion kicking in)
rows_fan = []
for horizon_m in range(1, 25):
    for action, p in REGIME_PARAMS.items():
        delta = p["monthly_delta"] if horizon_m <= 12 else p["monthly_delta"] * 0.5
        stress_mean = max(0.0, p["initial_stress"] + delta * horizon_m)

        # Uncertainty bands widen with sqrt(t) × action-specific vol
        action_vol = {"Ease": 0.018, "Hold": 0.014, "Tighten": 0.026}[action]
        band = action_vol * math.sqrt(horizon_m)

        # Crisis probability rises monotonically, anchored at 12m value
        base_crisis = p["crisis_prob_12m"]
        # Scale linearly: 0 at t=0, base at t=12, slow saturation after
        crisis_prob = min(0.95, base_crisis * (horizon_m / 12) if horizon_m <= 12
                          else base_crisis + (0.95 - base_crisis) * (1 - math.exp(-(horizon_m - 12) / 24)))

        rows_fan.append({
            "horizon_months": horizon_m,
            "policy_action": action,
            "stress_mean": round(stress_mean, 4),
            "stress_p10": round(max(0.0, stress_mean - 1.282 * band), 4),
            "stress_p90": round(stress_mean + 1.282 * band, 4),
            "crisis_prob": round(crisis_prob, 4),
        })

fan_path = OUT_DIR / "policy_fanChart.csv"
with open(fan_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows_fan[0].keys())
    writer.writeheader()
    writer.writerows(rows_fan)
print(f"  Written {len(rows_fan)} rows → {fan_path}")


# ── 4. Build regime_policy_sensitivity.csv ───────────────────────────────────
# Shows policy leverage per regime at the 12-month horizon.
# Normal regime anchored to real API data.
# Fragile / Crisis extrapolated via model noise scales (1.8×, 3.0×).

normal_stress = {a: comparison[a]["mean_stress"] for a in ("Ease", "Hold", "Tighten")}
normal_crisis = {a: comparison[a]["crisis_end"]  for a in ("Ease", "Hold", "Tighten")}

# Noise scale multipliers from monte_carlo.py: Normal=1.0, Fragile=1.8, Crisis=3.0
NOISE_SCALES = {"Normal": 1.0, "Fragile": 1.8, "Crisis": 3.0}

# Baseline crisis prob for each regime (from Kalman EM regime occupancy estimates)
REGIME_BASELINE_CRISIS = {"Normal": 0.42, "Fragile": 0.62, "Crisis": 0.88}

rows_sensitivity = []
for regime, noise in NOISE_SCALES.items():
    for action in ("Ease", "Hold", "Tighten"):
        stress_12m = normal_stress[action] * noise
        crisis_12m = min(0.98,
                         REGIME_BASELINE_CRISIS[regime]
                         + (normal_crisis[action] - normal_crisis["Hold"]) * noise)
        rows_sensitivity.append({
            "regime": regime,
            "policy_action": action,
            "stress_delta_12m": round(stress_12m, 4),
            "crisis_prob_12m": round(crisis_12m, 4),
            "leverage_vs_hold": round(abs(stress_12m - normal_stress["Hold"] * noise), 4),
        })

sens_path = OUT_DIR / "regime_policy_sensitivity.csv"
with open(sens_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows_sensitivity[0].keys())
    writer.writeheader()
    writer.writerows(rows_sensitivity)
print(f"  Written {len(rows_sensitivity)} rows → {sens_path}")


# ── 5. Print summary ─────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────")
print(f"  Date:        {state['latest_date']}")
print(f"  Regime:      {state['regime_label']}")
print(f"  Stress:      {state['stress_score']:.3f}")
print(f"  Fed Funds:   {state['fed_rate']*100:.2f}%")
print(f"  Inflation:   {state['inflation_gap']*100:+.2f}% above target")
print(f"\n  Policy leverage (12m stress delta, Ease vs Tighten):")
for regime, noise in NOISE_SCALES.items():
    ease_s  = round(normal_stress["Ease"]    * noise, 4)
    tight_s = round(normal_stress["Tighten"] * noise, 4)
    gap     = round(abs(ease_s - tight_s), 4)
    print(f"    {regime:8s}: Ease={ease_s:+.3f}  Tighten={tight_s:+.3f}  Gap={gap:.3f}")
print(f"\n  Files saved to: {OUT_DIR}")
