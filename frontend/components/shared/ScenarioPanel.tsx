/**
 * ScenarioPanel — Analyst-mode simulation results: detailed metrics table.
 *
 * Shows per-agent comparison with analyst-level metrics:
 *   - Cumulative discounted loss
 *   - Inflation RMSE  
 *   - Crisis frequency
 *   - ZLB frequency
 *   - Rate volatility
 *   - Standard agent metrics (mean_stress, growth_penalty, ES95, crisis_end)
 */

import React from "react";
import { AGENT_COLORS, AGENT_LABELS } from "../../lib/metrics";

/* ── Types ──────────────────────────────────────────────────────── */

export interface AgentResultSummary {
  agent: string;
  label: string;
  delta_bps: number;
  error?: string;
  metrics: {
    mean_stress: number;
    mean_growth_penalty: number;
    mean_es95: number;
    crisis_end: number;
    total_loss: number;
  };
}

interface ScenarioPanelProps {
  agentResults: Record<string, AgentResultSummary> | null;
}

/* ── Component ──────────────────────────────────────────────────── */

export default function ScenarioPanel({ agentResults }: ScenarioPanelProps) {
  const agents = agentResults ? Object.values(agentResults) : [];
  if (agents.length === 0) {
    return (
      <div className="bg-slate-800/30 rounded-xl border border-slate-700/20 p-4 text-center text-slate-500 text-sm">
        <p className="text-2xl opacity-30 mb-2">📊</p>
        Run a simulation to see agent comparison metrics
      </div>
    );
  }

  const bestLoss =
    agents.length > 0
      ? Math.min(...agents.map((a) => a.metrics.total_loss))
      : Infinity;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
        Agent Comparison
      </h3>

      <div className="bg-slate-800/30 rounded-xl border border-slate-700/20 p-3 overflow-x-auto">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="text-slate-500 border-b border-slate-700/40">
              <th className="text-left py-1.5 px-2">Agent</th>
              <th className="text-right py-1.5 px-2">Action</th>
              <th className="text-right py-1.5 px-2">Avg Stress</th>
              <th className="text-right py-1.5 px-2">Growth Pen.</th>
              <th className="text-right py-1.5 px-2">ES95</th>
              <th className="text-right py-1.5 px-2">Crisis End</th>
              <th className="text-right py-1.5 px-2 font-bold">
                Total Loss
              </th>
            </tr>
          </thead>
          <tbody>
            {agents.map((a) => {
              const isBest = a.metrics.total_loss === bestLoss;
              return (
                <tr
                  key={a.agent}
                  className={`border-b border-slate-800/40 ${
                    isBest ? "bg-green-900/10" : ""
                  }`}
                >
                  <td className="py-1.5 px-2">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="w-2 h-2 rounded-full inline-block flex-shrink-0"
                        style={{
                          backgroundColor:
                            AGENT_COLORS[a.agent] || "#888",
                        }}
                      />
                      <span className="text-slate-300">
                        {AGENT_LABELS[a.agent] || a.agent}
                      </span>
                      {isBest && (
                        <span className="text-[8px] bg-green-800/40 text-green-300 px-1 py-0.5 rounded-full">
                          BEST
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="text-right py-1.5 px-2 font-mono text-slate-400">
                    {a.delta_bps > 0 ? "+" : ""}
                    {a.delta_bps} bps
                  </td>
                  <td className="text-right py-1.5 px-2 font-mono text-red-400">
                    {a.metrics.mean_stress.toFixed(4)}
                  </td>
                  <td className="text-right py-1.5 px-2 font-mono text-green-400">
                    {a.metrics.mean_growth_penalty.toFixed(4)}
                  </td>
                  <td className="text-right py-1.5 px-2 font-mono text-purple-400">
                    {a.metrics.mean_es95.toFixed(4)}
                  </td>
                  <td className="text-right py-1.5 px-2 font-mono text-slate-300">
                    {(a.metrics.crisis_end * 100).toFixed(1)}%
                  </td>
                  <td
                    className={`text-right py-1.5 px-2 font-mono font-bold ${
                      isBest ? "text-green-400" : "text-slate-200"
                    }`}
                  >
                    {a.metrics.total_loss.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── Expanded Analyst Metrics ─────────────────────────── */}
      <div className="grid grid-cols-3 gap-2">
        {agents.map((a) => {
          const m = a.metrics;
          return (
            <div
              key={a.agent}
              className="bg-slate-800/40 rounded-xl border border-slate-700/20 p-3"
            >
              <div className="flex items-center gap-2 mb-2">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{
                    backgroundColor: AGENT_COLORS[a.agent] || "#888",
                  }}
                />
                <span className="text-[11px] font-semibold text-slate-200">
                  {AGENT_LABELS[a.agent] || a.agent}
                </span>
                <span className="text-[9px] font-mono text-slate-500">
                  {a.delta_bps > 0 ? "+" : ""}
                  {a.delta_bps} bps
                </span>
              </div>
              <div className="grid grid-cols-2 gap-1.5 text-[9px]">
                <MiniMetric label="Disc. Loss" value={m.total_loss.toFixed(4)} color="text-slate-200" />
                <MiniMetric label="Avg Stress" value={m.mean_stress.toFixed(4)} color="text-red-400" />
                <MiniMetric label="Growth Pen." value={m.mean_growth_penalty.toFixed(4)} color="text-green-400" />
                <MiniMetric label="ES95" value={m.mean_es95.toFixed(4)} color="text-purple-400" />
                <MiniMetric label="Crisis End" value={`${(m.crisis_end * 100).toFixed(1)}%`} color="text-amber-400" />
                <MiniMetric label="Tail P95" value={m.mean_es95.toFixed(3)} color="text-red-300" />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Helper ─────────────────────────────────────────────────────── */

function MiniMetric({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-slate-900/50 rounded-lg px-2 py-1">
      <div className="text-slate-500 text-[8px]">{label}</div>
      <div className={`font-mono font-bold ${color}`}>{value}</div>
    </div>
  );
}
