/**
 * PolicyPanel — Agent selection + simulation controls.
 *
 * Agents: heuristic, rl.
 * Historical Fed policy is NOT selectable — it auto-shows for pre-2026 dates
 * as part of the historical trendline (same colour/style).
 */

import React from "react";
import {
  AGENT_COLORS,
  AGENT_LABELS,
  AGENT_ICONS,
  AGENT_DESCS,
} from "../../lib/metrics";

/* ── Types ──────────────────────────────────────────────────────── */

export interface SimParams {
  delta_bps: number;
  alpha: number;
  beta: number;
  gamma: number;
  lambda: number;
  N: number;
  H: number;
  shocks: { credit: number; vol: number; rate: number };
  regime_switching: boolean;
}

interface PolicyPanelProps {
  params: SimParams;
  setParams: React.Dispatch<React.SetStateAction<SimParams>>;
  enabledAgents: string[];
  onToggleAgent: (id: string) => void;
  onRun: () => void;
  onReset: () => void;
  isRunning: boolean;
  selectedDate: string | null;
}

/* ── Component ──────────────────────────────────────────────────── */

export default function PolicyPanel({
  params,
  setParams,
  enabledAgents,
  onToggleAgent,
  onRun,
  onReset,
  isRunning,
}: PolicyPanelProps) {
  return (
    <div className="space-y-3 p-3">
      {/* ── Header ────────────────────────────────────────────── */}
      <div className="flex items-center gap-2">
        <span className="text-base">🔬</span>
        <h2 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
          Simulation
        </h2>
        {isRunning && (
          <span className="flex items-center gap-1 text-[10px] text-indigo-400">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
            Running
          </span>
        )}
      </div>

      {/* ── Agent Selection ───────────────────────────────────── */}
      <Section title="Policy Agents" icon="🤖">
        {(["heuristic", "rl"] as const).map((id) => (
          <div key={id} className="mb-2">
            <label
              className={`flex items-center gap-2 px-2 py-1.5 rounded-lg transition-all text-[11px] cursor-pointer ${
                enabledAgents.includes(id)
                  ? "bg-slate-700/40 text-slate-100"
                  : "text-slate-500 hover:bg-slate-800/40"
              }`}
            >
              <input
                type="checkbox"
                checked={enabledAgents.includes(id)}
                onChange={() => onToggleAgent(id)}
                className="accent-indigo-500 w-3 h-3"
              />
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: AGENT_COLORS[id] }}
              />
              <span>
                {AGENT_ICONS[id]} {AGENT_LABELS[id]}
              </span>
            </label>

            {/* Heuristic weights */}
            {id === "heuristic" && enabledAgents.includes(id) && (
              <div className="pl-8 mt-1.5 space-y-1">
                {(
                  [
                    { key: "alpha" as const, label: "Stability (α)", color: "text-blue-400" },
                    { key: "beta" as const, label: "Growth (β)", color: "text-green-400" },
                    { key: "gamma" as const, label: "Tail Risk (γ)", color: "text-orange-400" },
                    { key: "lambda" as const, label: "Crisis Exit (λ)", color: "text-red-400" },
                  ] as const
                ).map(({ key, label, color }) => (
                  <div key={key}>
                    <div className="flex justify-between text-[9px] mb-0.5">
                      <span className={color}>{label}</span>
                      <span className="font-mono text-slate-400">
                        {params[key].toFixed(1)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={5}
                      step={0.1}
                      value={params[key]}
                      onChange={(e) =>
                        setParams((p) => ({
                          ...p,
                          [key]: Number(e.target.value),
                        }))
                      }
                      className="w-full h-1 accent-emerald-500"
                    />
                  </div>
                ))}
              </div>
            )}

            {/* RL info */}
            {id === "rl" && enabledAgents.includes(id) && (
              <div className="pl-8 mt-1 text-[9px] text-slate-500">
                Uses trained PPO neural policy
              </div>
            )}

            {!enabledAgents.includes(id) && (
              <div className="pl-8 text-[8px] text-slate-600 mt-0.5">
                {AGENT_DESCS[id]}
              </div>
            )}
          </div>
        ))}
      </Section>

      {/* ── Shock Injection ───────────────────────────────────── */}
      <Section title="Shock Injection" icon="💥">
        {(
          [
            { key: "credit" as const, label: "Credit", desc: "spread widening" },
            { key: "vol" as const, label: "Volatility", desc: "VIX spike" },
            { key: "rate" as const, label: "Rate", desc: "yield shift" },
          ] as const
        ).map(({ key, label, desc }) => (
          <div key={key} className="mb-1.5">
            <div className="flex justify-between text-[10px] mb-0.5">
              <span className="text-amber-400/80">
                {label}{" "}
                <span className="text-slate-600 text-[8px]">{desc}</span>
              </span>
              <span className="font-mono text-slate-400">
                {params.shocks[key].toFixed(1)}σ
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={3}
              step={0.1}
              value={params.shocks[key]}
              onChange={(e) =>
                setParams((p) => ({
                  ...p,
                  shocks: { ...p.shocks, [key]: Number(e.target.value) },
                }))
              }
              className="w-full h-1 accent-amber-500"
            />
          </div>
        ))}
      </Section>

      {/* ── Monte Carlo Settings ──────────────────────────────── */}
      <Section title="Monte Carlo" icon="⚙">
        <Slider
          label="Paths (N)"
          value={params.N}
          min={500}
          max={10000}
          step={500}
          onChange={(v) => setParams((p) => ({ ...p, N: v }))}
          fmt={(v) => v.toLocaleString()}
        />
        <Slider
          label="Horizon (months)"
          value={params.H}
          min={6}
          max={36}
          step={1}
          onChange={(v) => setParams((p) => ({ ...p, H: v }))}
        />
        <div className="flex items-center justify-between mt-1">
          <span className="text-[10px] text-slate-400">Regime Switching</span>
          <button
            onClick={() =>
              setParams((p) => ({
                ...p,
                regime_switching: !p.regime_switching,
              }))
            }
            className={`w-8 h-4 rounded-full transition-all relative ${
              params.regime_switching ? "bg-indigo-600" : "bg-slate-700"
            }`}
          >
            <span
              className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform ${
                params.regime_switching ? "left-[14px]" : "left-0.5"
              }`}
            />
          </button>
        </div>
      </Section>

      {/* ── Run / Reset ──────────────────────────────────────── */}
      <div className="space-y-1.5 pt-1">
        <button
          onClick={onRun}
          disabled={isRunning || enabledAgents.length === 0}
          className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-30 text-white px-2 py-2.5 rounded-lg text-[11px] font-semibold transition-all shadow-md shadow-indigo-900/20"
        >
          {isRunning ? "⏳ Running Simulation..." : "▶ Run Simulation"}
        </button>
        <button
          onClick={onReset}
          className="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 px-2 py-2 rounded-lg text-[11px] font-medium border border-slate-700/50 transition-all"
        >
          ↺ Reset Results
        </button>
        {enabledAgents.length === 0 && (
          <p className="text-[9px] text-slate-600 text-center">
            Select at least one agent to run
          </p>
        )}
      </div>
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────────────── */

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-slate-800/30 rounded-xl border border-slate-700/20 p-3 space-y-2">
      <div className="flex items-center gap-1.5">
        <span className="text-xs">{icon}</span>
        <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
          {title}
        </h3>
      </div>
      {children}
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  fmt,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}) {
  return (
    <div className="mb-1.5">
      <div className="flex justify-between text-[10px] mb-0.5">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-400">
          {fmt ? fmt(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1 accent-indigo-500"
      />
    </div>
  );
}
