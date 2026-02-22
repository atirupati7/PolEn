/**
 * AdvancedAnalytics — Research-mode structural diagnostics.
 *
 * Compact layout:
 *   Row 1: Correlation Matrix (small) | Eigenvalue Spectrum (small)
 *   Row 2: Latent State | MSI Decomposition
 *   Row 3: Fed Policy Detail (collapsed by default)
 */

import React, { useState } from "react";
import CorrelationHeatmap from "../CorrelationHeatmap";
import EigenSpectrum from "../EigenSpectrum";
import { regimeBg, computeCrisisProb } from "../../lib/metrics";
import {
  computeMSI,
  computePolicyStance,
  DEFAULT_MSI_WEIGHTS,
} from "../../lib/macroStability";

interface AdvancedAnalyticsProps {
  correlationMatrix: number[][];
  correlationLabels: string[];
  eigenvalues: number[];
  mu_T: number[];
  stressScore: number;
  regime: string;
  inflationGap: number;
  fedRate: number;
  /** AI-generated insight text */
  aiInsight?: string | null;
}

export default function AdvancedAnalytics({
  correlationMatrix,
  correlationLabels,
  eigenvalues,
  mu_T,
  stressScore,
  regime,
  inflationGap,
  fedRate,
  aiInsight,
}: AdvancedAnalyticsProps) {
  const growthFactor = mu_T[2] || 0;
  const crisisProb = computeCrisisProb(stressScore);
  const msi = computeMSI(inflationGap, stressScore, crisisProb, growthFactor);
  const stance = computePolicyStance(fedRate, inflationGap, growthFactor);

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
        Structural Diagnostics
      </h3>

      {/* ── AI Insight Panel (if available) ───────────────────── */}
      {aiInsight && (
        <div className="bg-indigo-950/40 border border-indigo-500/20 rounded-xl p-3">
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-sm">✨</span>
            <h4 className="text-[10px] font-bold text-indigo-300 uppercase tracking-wider">
              AI Analysis
            </h4>
          </div>
          <p className="text-[11px] text-slate-300 leading-relaxed whitespace-pre-line">
            {aiInsight}
          </p>
        </div>
      )}

      {/* ── Row 1: Correlation + Eigenvalue (compact, side by side) */}
      <div className="grid grid-cols-2 gap-3">
        <CollapsibleCard title="Correlation Matrix" defaultOpen={false}>
          {correlationMatrix.length > 0 ? (
            <div className="max-h-48 overflow-auto">
              <CorrelationHeatmap
                matrix={correlationMatrix}
                labels={correlationLabels}
              />
            </div>
          ) : (
            <EmptyState text="No correlation data" />
          )}
        </CollapsibleCard>

        <CollapsibleCard title="Eigenvalue Spectrum" defaultOpen={false}>
          {eigenvalues.length > 0 ? (
            <div className="max-h-48 overflow-auto">
              <EigenSpectrum
                eigenvalues={eigenvalues}
                labels={correlationLabels}
              />
            </div>
          ) : (
            <EmptyState text="No eigenvalue data" />
          )}
        </CollapsibleCard>
      </div>

      {/* ── Row 2: Latent State + MSI ────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        <CollapsibleCard title="Latent State (μ_T)" defaultOpen>
          <div className="space-y-2">
            {[
              { label: "Stress", value: mu_T[0], color: "#ef4444", icon: "📉" },
              { label: "Liquidity", value: mu_T[1], color: "#3b82f6", icon: "💧" },
              { label: "Growth", value: mu_T[2], color: "#22c55e", icon: "📈" },
              { label: "Infl Gap", value: inflationGap, color: "#f59e0b", icon: "🔥" },
            ].map((s) => (
              <div key={s.label} className="flex items-center gap-2">
                <span className="text-xs w-4 text-center">{s.icon}</span>
                <div className="flex-1">
                  <div className="flex justify-between text-[9px]">
                    <span className="text-slate-500">{s.label}</span>
                    <span
                      className="font-mono font-bold"
                      style={{ color: s.color }}
                    >
                      {(s.value ?? 0).toFixed(4)}
                    </span>
                  </div>
                  <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden mt-0.5">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.min(
                          (Math.abs(s.value ?? 0) / 3) * 100,
                          100,
                        )}%`,
                        backgroundColor: s.color,
                        opacity: 0.7,
                      }}
                    />
                  </div>
                </div>
              </div>
            ))}

            <div className="bg-slate-900/60 rounded-lg p-1.5 text-center mt-1">
              <div className="text-[8px] text-slate-500">Regime</div>
              <span
                className={`inline-block text-[9px] font-bold px-1.5 py-0.5 rounded ${regimeBg(regime)}`}
              >
                {regime}
              </span>
            </div>
          </div>
        </CollapsibleCard>

        <CollapsibleCard title="MSI Decomposition" defaultOpen>
          <div className="space-y-2">
            <div className="text-center mb-1">
              <div className="text-2xl font-bold font-mono" style={{
                color: msi.score >= 75 ? '#22c55e' : msi.score >= 50 ? '#4ade80' : msi.score >= 30 ? '#f59e0b' : '#ef4444'
              }}>
                {msi.score}
              </div>
              <div className={`text-[9px] font-semibold ${msi.color}`}>
                {msi.label}
              </div>
            </div>

            {[
              { label: "Inflation", value: msi.components.inflation, weight: DEFAULT_MSI_WEIGHTS.inflation, color: "#f59e0b" },
              { label: "Stress", value: msi.components.stress, weight: DEFAULT_MSI_WEIGHTS.stress, color: "#ef4444" },
              { label: "Crisis", value: msi.components.crisis, weight: DEFAULT_MSI_WEIGHTS.crisis, color: "#a855f7" },
              { label: "Growth Vol", value: msi.components.growthVol, weight: DEFAULT_MSI_WEIGHTS.growthVol, color: "#22c55e" },
            ].map((c) => (
              <div key={c.label}>
                <div className="flex justify-between text-[8px] mb-0.5">
                  <span className="text-slate-500">
                    {c.label}{" "}
                    <span className="text-slate-600">
                      (w={c.weight.toFixed(2)})
                    </span>
                  </span>
                  <span className="font-mono font-bold" style={{ color: c.color }}>
                    {(c.value * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${c.value * 100}%`, backgroundColor: c.color, opacity: 0.7 }}
                  />
                </div>
              </div>
            ))}
          </div>
        </CollapsibleCard>
      </div>

      {/* ── Fed Policy Detail ─────────────────────────────────── */}
      <CollapsibleCard title="Fed Policy Detail" defaultOpen={false}>
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-slate-900/60 rounded-lg p-2 text-center">
            <div className="text-[8px] text-slate-500">Fed Funds Rate</div>
            <div className="text-sm font-mono font-bold text-blue-400">
              {(fedRate * 100).toFixed(2)}%
            </div>
          </div>
          <div className="bg-slate-900/60 rounded-lg p-2 text-center">
            <div className="text-[8px] text-slate-500">Taylor Rule</div>
            <div className="text-sm font-mono font-bold text-purple-400">
              {(stance.taylorRate * 100).toFixed(2)}%
            </div>
          </div>
          <div className="bg-slate-900/60 rounded-lg p-2 text-center">
            <div className="text-[8px] text-slate-500">Deviation</div>
            <div className={`text-sm font-mono font-bold ${stance.color}`}>
              {stance.deviationBps > 0 ? "+" : ""}
              {stance.deviationBps} bps
            </div>
          </div>
        </div>
        <div className="bg-slate-900/60 rounded-lg p-2 mt-2 text-center">
          <div className={`text-xs font-bold ${stance.color}`}>
            {stance.icon} {stance.category}
          </div>
          <div className="text-[8px] text-slate-500 mt-0.5">
            {stance.description}
          </div>
        </div>
      </CollapsibleCard>
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────────────── */

function CollapsibleCard({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="bg-slate-800/40 rounded-xl border border-slate-700/30 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-1.5 hover:bg-slate-700/20 transition-colors"
      >
        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
          {title}
        </h4>
        <span className="text-slate-500 text-xs transition-transform duration-200"
          style={{ transform: open ? 'rotate(0deg)' : 'rotate(-90deg)' }}>
          ▼
        </span>
      </button>
      {open && <div className="px-3 pb-2">{children}</div>}
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="text-center text-slate-500 text-[10px] py-4">{text}</div>
  );
}
