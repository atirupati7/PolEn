/**
 * PolEn — Policy Engine Control Room
 *
 * Single-view research interface:
 *   ┌──────────────────────────────────────────────────────────┐
 *   │ Header  [PolEn]                              [Refresh]   │
 *   ├──────────────────────────────────────────────────────────┤
 *   │ ▸ Recommended Policy Action Banner                       │
 *   ├──────────────────────────────────────────────────────────┤
 *   │ Date Slider                                              │
 *   ├────────────────────────────────────────────┬─────────────┤
 *   │                                            │             │
 *   │  5 Cards + 2×2 Charts + Comparison         │  Policy     │
 *   │  + Structural Diagnostics                  │  Panel      │
 *   │                                            │             │
 *   └────────────────────────────────────────────┴─────────────┘
 */

import React, { useState, useEffect, useCallback, useMemo } from "react";
import Head from "next/head";

import DateSlider from "../components/DateSlider";
import PolicyPanel from "../components/shared/PolicyPanel";
import ResearchMode from "../components/modes/ResearchMode";

import type { AgentResult, GeminiAdjustments } from "../components/shared/Charts";
import type { SimParams } from "../components/shared/PolicyPanel";
import { computePolicyStance } from "../lib/macroStability";

/* ── Constants ──────────────────────────────────────────────────── */

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ── Types ──────────────────────────────────────────────────────── */

export interface MacroState {
  latest_date: string;
  is_synthetic: boolean;
  mu_T: number[];
  P_T: number[][];
  stress_score: number;
  regime_label: string;
  crisis_threshold: number;
  inflation_gap?: number;
  fed_rate?: number;
  metrics: Record<string, number>;
  correlation_matrix: number[][];
  correlation_labels: string[];
  eigenvalues: number[];
}

/* ── Page ────────────────────────────────────────────────────────── */

export default function Home() {
  /* ── Data state ─────────────────────────────────────────────── */
  const [macroState, setMacroState] = useState<MacroState | null>(null);
  const [timeseries, setTimeseries] = useState<any>(null);
  const [historicalDates, setHistoricalDates] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /* ── Date selection ─────────────────────────────────────────── */
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [snapshotState, setSnapshotState] = useState<MacroState | null>(null);

  /* ── Agent state ────────────────────────────────────────────── */
  const [enabledAgents, setEnabledAgents] = useState<string[]>(["heuristic"]);
  const [agentResults, setAgentResults] = useState<Record<string, AgentResult> | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  /* ── Gemini AI state ────────────────────────────────────────── */
  const [aiInsight, setAiInsight] = useState<string | null>(null);  const [geminiRecommendation, setGeminiRecommendation] = useState<string | null>(null);
  const [geminiAdjustments, setGeminiAdjustments] = useState<GeminiAdjustments | null>(null);
  /* ── Simulation parameters ──────────────────────────────────── */
  const [params, setParams] = useState<SimParams>({
    delta_bps: 0,
    alpha: 1.0,
    beta: 1.0,
    gamma: 1.0,
    lambda: 1.0,
    N: 3000,
    H: 24,
    shocks: { credit: 0, vol: 0, rate: 0 },
    regime_switching: true,
  });

  /* ── Derived ────────────────────────────────────────────────── */
  const displayState: MacroState | null = snapshotState ?? macroState;
  const stressScore = displayState?.stress_score ?? 0;
  const growthFactor = displayState?.mu_T?.[2] ?? 0;
  const inflationGap = displayState?.inflation_gap ?? 0;
  const fedRate = displayState?.fed_rate ?? 0.03;
  const regime = displayState?.regime_label ?? "Unknown";

  /* ── Recommended policy (Taylor rule based) ─────────────────── */
  const recommendation = useMemo(() => {
    if (!displayState) return null;
    const stance = computePolicyStance(fedRate, inflationGap, growthFactor);
    // Recommendation is the opposite of deviation: if policy is too tight, ease
    // Clamp to a realistic monetary-policy range (Fed moves in 25 bps increments)
    const rawBps = -stance.deviationBps;
    const adjustBps = Math.max(-200, Math.min(200, rawBps));
    let action: string;
    let actionColor: string;
    let actionIcon: string;
    if (adjustBps < -10) {
      action = "Tighten";
      actionColor = "text-red-400 border-red-500/30 bg-red-950/40";
      actionIcon = "🔺";
    } else if (adjustBps > 10) {
      action = "Ease";
      actionColor = "text-green-400 border-green-500/30 bg-green-950/40";
      actionIcon = "🔻";
    } else {
      action = "Hold";
      actionColor = "text-slate-300 border-slate-500/30 bg-slate-900/40";
      actionIcon = "⏸";
    }
    return {
      action,
      bps: adjustBps,
      taylorRate: stance.taylorRate,
      color: actionColor,
      icon: actionIcon,
      description: stance.description,
    };
  }, [displayState, fedRate, inflationGap, growthFactor]);

  /* ═══════════════════════════════════════════════════════════════
     Effects
     ═══════════════════════════════════════════════════════════════ */

  useEffect(() => {
    refreshState();
  }, []);

  useEffect(() => {
    if (macroState) {
      fetchTimeseries();
      fetchDates();
    }
  }, [macroState]);

  useEffect(() => {
    if (selectedDate) fetchSnapshot(selectedDate);
    else setSnapshotState(null);
  }, [selectedDate]);

  // Gemini integration disabled — using pure MC-based agents
  // useEffect(() => {
  //   if (displayState && !agentResults) fetchGeminiInsight(null);
  // }, [displayState?.latest_date]);

  /* ═══════════════════════════════════════════════════════════════
     API Helpers
     ═══════════════════════════════════════════════════════════════ */

  const refreshState = async () => {
    setLoading(true);
    setError(null);
    try {
      const refreshRes = await fetch(`${API}/api/state/refresh`, { method: "POST" });
      if (!refreshRes.ok) {
        const err = await refreshRes.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to refresh state");
      }
      const stateRes = await fetch(`${API}/api/state/current`);
      if (!stateRes.ok) throw new Error("Failed to get current state");
      setMacroState(await stateRes.json());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchTimeseries = async () => {
    try {
      const res = await fetch(`${API}/api/historical/timeseries?years=30`);
      if (res.ok) setTimeseries(await res.json());
    } catch (e) {
      console.error("Failed to fetch timeseries:", e);
    }
  };

  const fetchDates = async () => {
    try {
      const res = await fetch(`${API}/api/historical/dates`);
      if (res.ok) {
        const data = await res.json();
        setHistoricalDates(data.dates || []);
      }
    } catch (e) {
      console.error("Failed to fetch dates:", e);
    }
  };

  const fetchSnapshot = async (date: string) => {
    try {
      const res = await fetch(`${API}/api/historical/state?date=${date}`);
      if (!res.ok) return;
      const d = await res.json();
      setSnapshotState({
        latest_date: d.date || d.latest_date || date,
        is_synthetic: false,
        mu_T: d.mu_T || [],
        P_T: d.P_T || [[]],
        stress_score: d.stress_score ?? 0,
        regime_label: d.regime_label || "Unknown",
        crisis_threshold: d.crisis_threshold ?? 0,
        inflation_gap: d.inflation_gap,
        fed_rate: d.fed_rate,
        metrics: d.metrics || {},
        correlation_matrix: d.correlation_matrix || [],
        correlation_labels: d.correlation_labels || [],
        eigenvalues: d.eigenvalues || [],
      });
    } catch (e) {
      console.error("Failed to fetch snapshot:", e);
    }
  };

  /* ═══════════════════════════════════════════════════════════════
     Handlers
     ═══════════════════════════════════════════════════════════════ */

  /** Fetch Gemini AI insight for current macro state */
  const fetchGeminiInsight = useCallback(
    async (results: Record<string, AgentResult> | null) => {
      if (!displayState) return;
      try {
        const body: Record<string, any> = {
          stress_score: stressScore,
          growth_factor: growthFactor,
          inflation_gap: inflationGap,
          fed_rate: fedRate,
          regime,
          crisis_probability: stressScore > (displayState.crisis_threshold ?? 1) ? 0.7 : 0.15,
          msi_score: Math.max(0, Math.min(100, Math.round(50 + growthFactor * 20 - stressScore * 15))),
          selected_date: selectedDate || displayState.latest_date,
          recommendation_action: recommendation?.action,
          recommendation_bps: recommendation?.bps,
        };
        if (results) {
          body.agent_results = Object.fromEntries(
            Object.entries(results).map(([k, v]) => [
              k,
              { delta_bps: v.delta_bps, metrics: v.metrics },
            ]),
          );
        }
        const res = await fetch(`${API}/api/gemini/enhance`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (res.ok) {
          const data = await res.json();
          // Combine narrative fields into a single insight paragraph
          const parts = [data.insight, data.model_vs_fed, data.risk_narrative].filter(Boolean);
          setAiInsight(parts.join(" "));
          // Store enhanced recommendation separately for the banner
          if (data.enhanced_recommendation) setGeminiRecommendation(data.enhanced_recommendation);
          // Store trajectory adjustments for Charts
          if (data.trajectory_adjustments) {
            setGeminiAdjustments(data.trajectory_adjustments);
          }
        }
      } catch (e) {
        console.error("Gemini insight error:", e);
      }
    },
    [displayState, stressScore, growthFactor, inflationGap, fedRate, regime, selectedDate, recommendation],
  );

  const handleRun = useCallback(async () => {
    setIsRunning(true);
    setError(null);
    setAgentResults(null);
    try {
      const body: Record<string, any> = {
        agents: Array.from(new Set([...enabledAgents, "historical"])),
        custom_delta_bps: params.delta_bps,
        alpha: params.alpha,
        beta: params.beta,
        gamma: params.gamma,
        lambda: params.lambda,
        N: params.N,
        H: params.H,
        regime_switching: params.regime_switching,
        shocks: params.shocks,
      };
      if (selectedDate) body.start_date = selectedDate;

      const res = await fetch(`${API}/api/agents/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Simulation failed");
      }
      const result = await res.json();
      setAgentResults(result.agents);
      // Gemini integration disabled — using pure MC-based agents
      // fetchGeminiInsight(result.agents);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsRunning(false);
    }
  }, [enabledAgents, params, selectedDate]);

  const handleReset = useCallback(() => {
    setAgentResults(null);
    setAiInsight(null);
    setGeminiRecommendation(null);
    setGeminiAdjustments(null);
  }, []);

  const handleDateChange = useCallback((date: string | null) => {
    setSelectedDate(date);
    setAgentResults(null);
  }, []);

  const toggleAgent = useCallback((id: string) => {
    setEnabledAgents((prev) =>
      prev.includes(id) ? prev.filter((a) => a !== id) : [...prev, id],
    );
  }, []);

  /* ═══════════════════════════════════════════════════════════════
     Render
     ═══════════════════════════════════════════════════════════════ */

  return (
    <>
      <Head>
        <title>PolEn | Policy Engine</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
        <div className="sticky top-0 z-40 bg-slate-950/95 backdrop-blur supports-[backdrop-filter]:bg-slate-950/80">
          {/* ── Header ──────────────────────────────────────────── */}
          <header className="bg-slate-900 border-b border-slate-700/50 px-6 py-2 flex items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-4">
              {/* Logo */}
              <h1 className="text-lg font-black tracking-tight">
                <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  PolEn
                </span>
                <span className="text-slate-500 font-normal text-sm ml-2">
                  Policy Engine
                </span>
              </h1>

              {/* Status badges */}
              {macroState?.is_synthetic && (
                <span className="text-[10px] bg-amber-800/60 text-amber-200 px-2 py-0.5 rounded-full border border-amber-700/40 font-medium">
                  SYNTHETIC
                </span>
              )}
              {displayState && (
                <span className="text-[10px] text-slate-500 font-mono">
                  {displayState.latest_date}
                </span>
              )}
            </div>

            <button
              onClick={refreshState}
              disabled={loading}
              className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-40 px-4 py-1.5 rounded-lg text-white text-xs font-medium transition-all shadow-md shadow-indigo-900/20"
            >
              {loading ? "↻ Loading..." : "↻ Refresh Data"}
            </button>
          </header>

          {/* ── Recommended Policy Banner ───────────────────────── */}
          {recommendation && displayState && (
            <div
              className={`border-b px-6 py-2.5 flex items-center justify-between flex-shrink-0 ${recommendation.color}`}
            >
              <div className="flex items-center gap-3">
                <span className="text-lg">{recommendation.icon}</span>
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider">
                    Recommended Action:{" "}
                  </span>
                  <span className="text-sm font-black">
                    {recommendation.action}
                    {recommendation.action !== "Hold" && (
                      <span className="ml-1.5 font-mono">
                        {recommendation.bps > 0 ? "−" : "+"}
                        {Math.abs(recommendation.bps)} bps
                      </span>
                    )}
                  </span>
                  {/* Gemini recommendation disabled — pure MC mode */}
                </div>
              </div>
              <div className="text-[10px] text-right opacity-70">
                <div>
                  Taylor-implied rate:{" "}
                  <span className="font-mono font-bold">
                    {(recommendation.taylorRate * 100).toFixed(2)}%
                  </span>
                </div>
                <div>
                  Current Fed rate:{" "}
                  <span className="font-mono font-bold">
                    {(fedRate * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* ── Error Banner ────────────────────────────────────── */}
          {error && (
            <div className="bg-red-900/40 border-b border-red-800/50 px-6 py-2 text-red-200 text-sm flex items-center justify-between flex-shrink-0">
              <span className="flex items-center gap-2">
                <span className="text-red-400">⚠</span> {error}
              </span>
              <button
                onClick={() => setError(null)}
                className="text-red-400 hover:text-red-200 transition-colors"
              >
                ✕
              </button>
            </div>
          )}

          {/* ── Date Slider ─────────────────────────────────────── */}
          {historicalDates.length > 0 && (
            <DateSlider
              dates={historicalDates}
              selectedDate={selectedDate}
              onDateChange={handleDateChange}
              regimeAtDate={displayState?.regime_label}
            />
          )}
        </div>

        {/* ── Main Layout ─────────────────────────────────────── */}
        <div className="flex flex-1 min-h-0">
          {/* Scrollable content */}
          <div className="flex-1 min-w-0 overflow-y-auto p-5 scrollbar-thin">
            {/* Empty state */}
            {!macroState && !loading && (
              <div className="flex items-center justify-center h-96 text-slate-400">
                <div className="text-center space-y-3">
                  <div className="text-5xl opacity-30">🌐</div>
                  <p className="text-xl font-semibold text-slate-300">
                    No data loaded
                  </p>
                  <p className="text-sm">
                    Click{" "}
                    <span className="text-indigo-400 font-medium">
                      Refresh Data
                    </span>{" "}
                    to initialize the pipeline.
                  </p>
                </div>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div className="flex items-center justify-center h-96 text-slate-400">
                <div className="text-center space-y-3">
                  <div className="animate-spin text-4xl">⚙</div>
                  <p className="text-sm">
                    Running data pipeline &amp; Kalman filter...
                  </p>
                </div>
              </div>
            )}

            {/* Data loaded → Research view */}
            {macroState && !loading && displayState && (
              <ResearchMode
                stressScore={stressScore}
                growthFactor={growthFactor}
                inflationGap={inflationGap}
                fedRate={fedRate}
                regime={regime}
                timeseries={timeseries}
                selectedDate={selectedDate}
                agentResults={agentResults}
                enabledAgents={Array.from(new Set([...enabledAgents, "historical"]))}
                correlationMatrix={displayState.correlation_matrix}
                correlationLabels={displayState.correlation_labels}
                eigenvalues={displayState.eigenvalues}
                mu_T={displayState.mu_T}
                dateCursor={selectedDate}
                aiInsight={aiInsight}
                geminiAdjustments={geminiAdjustments}
              />
            )}
          </div>

          {/* Right sidebar */}
          <aside className="w-72 flex-shrink-0 bg-slate-900/80 border-l border-slate-700/30 overflow-y-auto scrollbar-thin">
            <PolicyPanel
              params={params}
              setParams={setParams}
              enabledAgents={enabledAgents}
              onToggleAgent={toggleAgent}
              onRun={handleRun}
              onReset={handleReset}
              isRunning={isRunning}
              selectedDate={selectedDate}
            />
          </aside>
        </div>
      </div>
    </>
  );
}
