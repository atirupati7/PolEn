/**
 * MacroCards — Executive-level macro indicator cards.
 *
 * Displays exactly 5 core metrics + Macro Stability Index (MSI) gauge:
 *   1. Macro Stability Index (large radial gauge, 0–100)
 *   2. Inflation Gap
 *   3. Financial Stress
 *   4. Crisis Probability (large gauge)
 *   5. Policy Stance (Accommodative / Neutral / Restrictive)
 *
 * ≤7 numeric values visible, per UX spec.
 */

import React from "react";
import {
  computeMSI,
  computePolicyStance,
  DEFAULT_MSI_WEIGHTS,
} from "../../lib/macroStability";
import { computeCrisisProb, regimeBg } from "../../lib/metrics";

interface MacroCardsProps {
  stressScore: number;
  growthFactor: number;
  inflationGap: number;
  fedRate: number;
  regime: string;
}

export default function MacroCards({
  stressScore,
  growthFactor,
  inflationGap,
  fedRate,
  regime,
}: MacroCardsProps) {
  const crisisProb = computeCrisisProb(stressScore);
  const msi = computeMSI(
    inflationGap,
    stressScore,
    crisisProb,
    growthFactor,
    DEFAULT_MSI_WEIGHTS,
  );
  const stance = computePolicyStance(fedRate, inflationGap, growthFactor);

  return (
    <div className="grid grid-cols-5 gap-3">
      {/* ── 1. MSI Gauge (large) ─────────────────────────────── */}
      <div className="col-span-1 bg-slate-800/50 rounded-2xl border border-slate-700/30 p-4 flex flex-col items-center justify-center">
        <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">
          Macro Stability
        </div>
        <GaugeRing value={msi.score} color={msi.color} />
        <div className={`text-[10px] font-semibold mt-1.5 ${msi.color}`}>
          {msi.label}
        </div>
        <span className={`text-[8px] mt-0.5 px-1.5 py-0.5 rounded ${regimeBg(regime)}`}>
          {regime}
        </span>
      </div>

      {/* ── 2. Inflation Gap ─────────────────────────────────── */}
      <MetricCard
        label="Inflation Gap"
        value={`${inflationGap >= 0 ? "+" : ""}${(inflationGap * 100).toFixed(2)}%`}
        severity={msi.components.inflation}
        color={
          Math.abs(inflationGap) > 0.02
            ? "text-red-400"
            : Math.abs(inflationGap) > 0.01
              ? "text-amber-400"
              : "text-green-400"
        }
        sub={
          inflationGap > 0
            ? "Above target"
            : inflationGap < 0
              ? "Below target"
              : "On target"
        }
      />

      {/* ── 3. Financial Stress ──────────────────────────────── */}
      <MetricCard
        label="Financial Stress"
        value={`${stressScore.toFixed(2)}σ`}
        severity={msi.components.stress}
        color={
          stressScore > 1.5
            ? "text-red-400"
            : stressScore > 0.5
              ? "text-amber-400"
              : "text-green-400"
        }
        sub={
          stressScore > 1.5
            ? "Severe"
            : stressScore > 0.5
              ? "Elevated"
              : "Normal"
        }
      />

      {/* ── 4. Crisis Probability (large gauge) ──────────────── */}
      <div className="bg-slate-800/50 rounded-2xl border border-slate-700/30 p-4 flex flex-col items-center justify-center">
        <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">
          Crisis Probability
        </div>
        <GaugeRing
          value={Math.round(crisisProb * 100)}
          color={
            crisisProb > 0.3
              ? "text-red-400"
              : crisisProb > 0.1
                ? "text-amber-400"
                : "text-green-400"
          }
          suffix="%"
          invert
        />
        <div className="text-[10px] text-slate-500 mt-1">
          {crisisProb > 0.3 ? "High" : crisisProb > 0.1 ? "Moderate" : "Low"} risk
        </div>
      </div>

      {/* ── 5. Policy Stance ─────────────────────────────────── */}
      <div className="bg-slate-800/50 rounded-2xl border border-slate-700/30 p-4 flex flex-col items-center justify-center text-center">
        <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">
          Policy Stance
        </div>
        <div className={`text-2xl ${stance.color}`}>{stance.icon}</div>
        <div className={`text-sm font-bold mt-1 ${stance.color}`}>
          {stance.category}
        </div>
        <div className="text-[9px] font-mono text-slate-500 mt-0.5">
          {stance.deviationBps > 0 ? "+" : ""}
          {stance.deviationBps} bps vs Taylor
        </div>
        <div className="text-[8px] text-slate-600 mt-1 leading-tight">
          Fed {(fedRate * 100).toFixed(1)}% · Taylor{" "}
          {(stance.taylorRate * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────────────── */

function MetricCard({
  label,
  value,
  severity,
  color,
  sub,
}: {
  label: string;
  value: string;
  severity: number;
  color: string;
  sub: string;
}) {
  return (
    <div className="bg-slate-800/50 rounded-2xl border border-slate-700/30 p-4 flex flex-col items-center justify-center text-center">
      <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">
        {label}
      </div>
      <div className={`text-xl font-mono font-bold ${color}`}>{value}</div>
      {/* Severity bar */}
      <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden mt-2">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${severity * 100}%`,
            background:
              severity > 0.6
                ? "#ef4444"
                : severity > 0.3
                  ? "#f59e0b"
                  : "#22c55e",
          }}
        />
      </div>
      <div className="text-[9px] text-slate-500 mt-1">{sub}</div>
    </div>
  );
}

/**
 * Radial gauge ring rendered with SVG.
 * `value` is 0–100. `invert` means higher is worse (e.g. crisis prob).
 */
function GaugeRing({
  value,
  color,
  suffix = "",
  invert = false,
}: {
  value: number;
  color: string;
  suffix?: string;
  invert?: boolean;
}) {
  const r = 32;
  const stroke = 6;
  const circumference = 2 * Math.PI * r;
  const pct = Math.min(Math.max(value, 0), 100) / 100;
  const offset = circumference * (1 - pct);

  // Color for arc
  let arcColor: string;
  if (invert) {
    arcColor = value > 30 ? "#ef4444" : value > 10 ? "#f59e0b" : "#22c55e";
  } else {
    arcColor = value >= 75 ? "#22c55e" : value >= 50 ? "#4ade80" : value >= 30 ? "#f59e0b" : "#ef4444";
  }

  return (
    <div className="relative w-20 h-20 flex items-center justify-center">
      <svg
        width={80}
        height={80}
        viewBox={`0 0 ${(r + stroke) * 2} ${(r + stroke) * 2}`}
        className="-rotate-90"
      >
        {/* Background ring */}
        <circle
          cx={r + stroke}
          cy={r + stroke}
          r={r}
          fill="none"
          stroke="#1e293b"
          strokeWidth={stroke}
        />
        {/* Value arc */}
        <circle
          cx={r + stroke}
          cy={r + stroke}
          r={r}
          fill="none"
          stroke={arcColor}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-700 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className={`text-lg font-bold font-mono ${color}`}>
          {value}
          <span className="text-[10px]">{suffix}</span>
        </span>
      </div>
    </div>
  );
}
