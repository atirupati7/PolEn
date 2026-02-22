"""
MacroState Control Room — Streamlit Frontend.

Provides an interactive dashboard with:
- Policy engine mode toggle (Heuristic / RL)
- Current macro-state display
- Policy recommendation with comparison table
- RL vs Heuristic evaluation charts
- Action trajectory visualisation

Run:
    streamlit run streamlit_app.py
    # or from backend/:
    streamlit run streamlit_app.py -- --backend http://localhost:8001
"""

import argparse
import json

import requests
import streamlit as st
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8001"


def api(path: str, method: str = "GET", **kwargs):
    """Call backend API."""
    url = f"{BACKEND_URL}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=60)
        else:
            r = requests.post(url, json=kwargs.get("json", {}), timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {BACKEND_URL}. Is it running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Page config ─────────────────────────────────────────────────

st.set_page_config(
    page_title="MacroState Control Room",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ MacroState Control Room")
st.caption("Cross-Asset Macro State + Policy Engine")

# ── Sidebar: mode toggle & controls ────────────────────────────

st.sidebar.header("⚙️ Policy Engine")

# Fetch current mode
current_mode_resp = api("/api/policy/mode")
current_mode = current_mode_resp.get("mode", "heuristic") if current_mode_resp else "heuristic"

mode = st.sidebar.radio(
    "Policy Engine Mode",
    options=["heuristic", "rl"],
    index=0 if current_mode == "heuristic" else 1,
    format_func=lambda m: "🔧 Heuristic Optimizer" if m == "heuristic" else "🤖 RL Agent (PPO)",
)

# Update mode on backend if changed
if mode != current_mode:
    resp = api("/api/policy/set_mode", method="POST", json={"mode": mode})
    if resp:
        st.sidebar.success(f"Switched to **{mode}** mode")
    st.rerun()

# Mode indicator
if mode == "rl":
    st.sidebar.markdown("### 🤖 AI Central Banker Active")
    st.sidebar.info("The PPO-trained RL agent is controlling policy recommendations.")
else:
    st.sidebar.markdown("### 🔧 Heuristic Optimizer Active")
    st.sidebar.info("Monte Carlo stochastic loss minimiser is active.")

st.sidebar.divider()

# ── Data controls ──────────────────────────────────────────────

st.sidebar.header("📊 Data")
use_synthetic = st.sidebar.checkbox("Demo Synthetic Mode", value=False)

if st.sidebar.button("🔄 Refresh State"):
    with st.spinner("Refreshing data pipeline + Kalman estimation…"):
        resp = api("/api/state/refresh", method="POST", json={"synthetic": use_synthetic})
    if resp and resp.get("status") == "ok":
        st.sidebar.success(f"State refreshed — {resp.get('regime_label', '?')}")
    elif resp:
        st.sidebar.error(f"Refresh failed: {resp}")

st.sidebar.divider()

# ── Recommendation controls ────────────────────────────────────

st.sidebar.header("🎯 Recommendation")
alpha = st.sidebar.slider("Stress weight (α)", 0.0, 5.0, 1.0, 0.1)
beta = st.sidebar.slider("Growth penalty (β)", 0.0, 5.0, 1.0, 0.1)
gamma = st.sidebar.slider("Tail risk (γ)", 0.0, 5.0, 1.0, 0.1)
lam = st.sidebar.slider("Crisis end (λ)", 0.0, 5.0, 1.0, 0.1)
N = st.sidebar.select_slider("MC Paths", options=[500, 1000, 2000, 5000, 10000], value=5000)
H = st.sidebar.slider("Horizon (months)", 6, 36, 24)

# ── Main area ──────────────────────────────────────────────────

tab_state, tab_recommend, tab_eval = st.tabs([
    "📈 Current State",
    "🎯 Recommendation",
    "⚖️ RL vs Heuristic",
])

# ── Tab 1: Current State ──────────────────────────────────────

with tab_state:
    state = api("/api/state/current")
    if state:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Regime", state.get("regime_label", "—"))
        col2.metric("Stress Score", f"{state.get('stress_score', 0):.3f}")
        col3.metric("Latest Date", state.get("latest_date", "—"))
        col4.metric("Synthetic", "Yes" if state.get("is_synthetic") else "No")

        st.subheader("Latent State μ_T")
        mu = state.get("mu_T", [0, 0, 0])
        df_state = pd.DataFrame(
            {"Dimension": ["Stress", "Liquidity", "Growth"], "Value": mu}
        )
        st.bar_chart(df_state.set_index("Dimension"))

        # Correlation heatmap
        st.subheader("Cross-Asset Correlation Matrix")
        corr = state.get("correlation_matrix")
        labels = state.get("correlation_labels", [])
        if corr and labels:
            import plotly.express as px

            fig = px.imshow(
                corr,
                x=labels,
                y=labels,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                text_auto=".2f",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Eigenvalues
        st.subheader("Eigenvalue Spectrum")
        eigvals = state.get("eigenvalues", [])
        if eigvals:
            st.bar_chart(pd.DataFrame({"λ": eigvals}, index=[f"λ{i+1}" for i in range(len(eigvals))]))

        # Metrics
        st.subheader("Structure Metrics")
        metrics = state.get("metrics", {})
        if metrics:
            st.json(metrics)
    else:
        st.warning("State not loaded. Click **Refresh State** in the sidebar.")

# ── Tab 2: Policy Recommendation ──────────────────────────────

with tab_recommend:
    if st.button("🚀 Run Recommendation", type="primary"):
        with st.spinner(f"Running recommendation ({mode} mode)…"):
            payload = {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "lambda": lam,
                "N": N,
                "H": H,
                "regime_switching": True,
            }
            rec = api("/api/policy/recommend", method="POST", json=payload)

        if rec:
            engine_used = rec.get("engine", mode)
            st.info(f"Engine: **{engine_used.upper()}**")

            if engine_used == "rl":
                # RL result
                st.success(f"### Recommendation: **{rec.get('action_label', '?')}** ({rec.get('delta_bps', 0):+.0f} bps)")
                st.metric("Raw Action", f"{rec.get('action_raw', 0):.3f}")
                st.metric("Regime", rec.get("regime", "—"))
            else:
                # Heuristic result
                recommended = rec.get("recommended_action", "?")
                bps = rec.get("recommended_bps", 0)
                st.success(f"### Recommendation: **{recommended}** ({bps:+d} bps)")
                st.write(rec.get("explanation", ""))

                # Comparison table
                comparison = rec.get("comparison", [])
                if comparison:
                    st.subheader("Policy Comparison")
                    df_comp = pd.DataFrame(comparison)
                    st.dataframe(df_comp, use_container_width=True)

                    # Bar chart of total loss
                    st.bar_chart(
                        df_comp.set_index("action")["total_loss"],
                    )
    else:
        st.info("Click **Run Recommendation** to evaluate policy options.")

# ── Tab 3: RL vs Heuristic Evaluation ─────────────────────────

with tab_eval:
    st.write(
        "Run a head-to-head evaluation: 20 episodes each for the RL agent "
        "and the heuristic optimizer in the neural-transition environment."
    )

    if st.button("⚖️ Run Evaluation", type="secondary"):
        with st.spinner("Evaluating (this may take a minute)…"):
            ev = api("/api/policy/evaluation")

        if ev:
            rl = ev.get("rl", {})
            heur = ev.get("heuristic", {})
            comp = ev.get("comparison_summary", {})

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🤖 RL Agent")
                st.metric("Avg Episode Reward", f"{rl.get('avg_episode_reward', 0):.2f}")
                st.metric("Crisis Frequency", f"{rl.get('crisis_frequency', 0):.1%}")
                st.metric("ES95 Stress", f"{rl.get('es95_stress', 0):.3f}")
                st.metric("Mean |Action| (bps)", f"{rl.get('mean_abs_action_bps', 0):.1f}")

            with col2:
                st.subheader("🔧 Heuristic")
                st.metric("Avg Episode Reward", f"{heur.get('avg_episode_reward', 0):.2f}")
                st.metric("Crisis Frequency", f"{heur.get('crisis_frequency', 0):.1%}")
                st.metric("ES95 Stress", f"{heur.get('es95_stress', 0):.3f}")
                st.metric("Mean |Action| (bps)", f"{heur.get('mean_abs_action_bps', 0):.1f}")

            st.divider()
            st.subheader("Summary")
            if comp.get("rl_better_reward"):
                st.success(f"RL agent achieved **higher reward** by {comp.get('reward_advantage', 0):+.2f}")
            else:
                st.warning(f"Heuristic achieved higher reward by {-comp.get('reward_advantage', 0):+.2f}")

            if comp.get("rl_fewer_crises"):
                st.success(f"RL agent had **fewer crises** (Δ = {comp.get('crisis_frequency_delta', 0):+.3f})")
            else:
                st.warning(f"Heuristic had fewer crises (Δ = {comp.get('crisis_frequency_delta', 0):+.3f})")

            # Comparison bar chart
            metrics_names = ["avg_episode_reward", "crisis_frequency", "es95_stress"]
            df_eval = pd.DataFrame({
                "Metric": metrics_names * 2,
                "Value": [rl.get(m, 0) for m in metrics_names] + [heur.get(m, 0) for m in metrics_names],
                "Agent": ["RL"] * len(metrics_names) + ["Heuristic"] * len(metrics_names),
            })
            try:
                import plotly.express as px
                fig = px.bar(df_eval, x="Metric", y="Value", color="Agent", barmode="group")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(df_eval)
    else:
        st.info("Click **Run Evaluation** to compare RL vs Heuristic performance.")
