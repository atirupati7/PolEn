/**
 * Charts — Unified time-series charts (full 2×2 grid).
 *
 * Key behaviours:
 *   - Shows only a 24-month window around the selected date (not full history).
 *   - Historical trendline continues seamlessly for dates before Feb 2026;
 *     there is NO separate "historical" agent colour or style — it's the same
 *     grey historical line.
 *   - Agent simulation overlays (heuristic, rl) draw with animation each time
 *     new results arrive.
 *   - First agent gets fan chart bands.
 */

import React, { useMemo, useState, useEffect } from "react";
import {
  Area,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Line,
} from "recharts";
import {
  AGENT_COLORS,
  AGENT_LABELS,
  TOOLTIP_STYLE,
  fmtDateShort,
  addMonthsEnd,
  computeTimeTicks,
} from "../../lib/metrics";

/* ── Re-export types for consumers ──────────────────────────────── */

export interface TimeseriesData {
  dates: string[];
  series: Record<string, { label: string; values: (number | null)[] }>;
  latent_factors: {
    stress: number[];
    liquidity?: number[];
    growth?: number[];
  };
  regime_labels: string[];
  crisis_probability: number[];
  stress_mean?: number;
  stress_std?: number;
  regime_threshold_crisis?: number;
}

export interface AgentResult {
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
  crisis_prob_path: number[];
  stress_path: number[];
  growth_path: number[];
  es95_path?: (number | null)[];
  stress_fan: {
    p5: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
  }[];
  growth_fan: {
    p5: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
  }[];
}

/* ── Gemini trajectory adjustments ───────────────────────────── */

export interface GeminiAdjustments {
  stress: number;  // e.g. -5 → reduce stress by 5%
  growth: number;  // e.g. +3 → boost growth by 3%
  crisis: number;  // e.g. -8 → reduce crisis prob by 8%
  es95: number;    // e.g. -4 → reduce tail risk by 4%
}

/* ── Props ──────────────────────────────────────────────────────── */

interface ChartsProps {
  timeseries: TimeseriesData | null;
  selectedDate: string | null;
  agentResults: Record<string, AgentResult> | null;
  enabledAgents: string[];
  /** "executive" shows combined Stress+Growth & Crisis; "full" shows 2×2 */
  layout: "executive" | "full";
  /** Date cursor from the DateSlider — shown as a vertical line on all charts */
  dateCursor?: string | null;
  /** Gemini-sourced adjustments to apply to non-historical agent trajectories */
  geminiAdjustments?: GeminiAdjustments | null;
}

/* ── Helpers ────────────────────────────────────────────────────── */

/** Subtract N months from a YYYY-MM-DD string and return YYYY-MM-DD */
function subtractMonths(dateStr: string, n: number): string {
  const [y, m] = dateStr.split("-").map(Number);
  const total = y * 12 + (m - 1) - n;
  const newYear = Math.floor(total / 12);
  const newMonth = (total % 12) + 1;
  return `${newYear}-${String(newMonth).padStart(2, "0")}-01`;
}

/* ── Component ──────────────────────────────────────────────────── */

export default function Charts({
  timeseries,
  selectedDate,
  agentResults,
  enabledAgents,
  layout,
  dateCursor,
  geminiAdjustments,
}: ChartsProps) {
  /* Animation key — bumps whenever agentResults changes to retrigger draw */
  const [animKey, setAnimKey] = useState(0);
  useEffect(() => {
    if (agentResults) setAnimKey((k) => k + 1);
  }, [agentResults]);

  const hasAgents =
    agentResults != null &&
    enabledAgents.some((a) => agentResults[a] != null);

  const simStart = useMemo(() => {
    if (selectedDate) return selectedDate;
    if (timeseries && timeseries.dates.length > 0)
      return timeseries.dates[timeseries.dates.length - 1];
    return null;
  }, [selectedDate, timeseries]);

  /* ── 24-month window boundaries ──────────────────────────────── */
  const windowStart = useMemo(() => {
    const anchor = selectedDate ?? (timeseries?.dates?.[timeseries.dates.length - 1] ?? null);
    if (!anchor) return null;
    return subtractMonths(anchor, 24);
  }, [selectedDate, timeseries]);

  /* ── Build chart data (windowed) ─────────────────────────────── */
  const { stressData, growthData, crisisData, es95Data, activeAgents } =
    useMemo(() => {
      const stress: Record<string, any>[] = [];
      const growth: Record<string, any>[] = [];
      const crisis: Record<string, any>[] = [];
      const es95: Record<string, any>[] = [];
      const agents: string[] = [];

      if (timeseries && windowStart) {
        const total = timeseries.dates.length;
        for (let i = 0; i < total; i++) {
          const date = timeseries.dates[i];
          // Only include dates within the 24-month window before the anchor
          if (date < windowStart) continue;
          if (hasAgents && simStart && date > simStart) continue;
          // If no agents, stop at selected date + a few months buffer
          if (!hasAgents && simStart && date > addMonthsEnd(simStart, 2)) continue;
          stress.push({
            date,
            hist: timeseries.latent_factors.stress[i] ?? null,
          });
          growth.push({
            date,
            hist: timeseries.latent_factors.growth?.[i] ?? null,
          });
          crisis.push({
            date,
            hist:
              timeseries.crisis_probability[i] != null
                ? timeseries.crisis_probability[i] * 100
                : null,
          });
          es95.push({ date });
        }
      }

      if (hasAgents && agentResults && simStart) {
        // Helper: apply Gemini percentage adjustment to a value
        const adj = (val: number | null | undefined, pct: number): number | null => {
          if (val == null) return null;
          return val * (1 + pct / 100);
        };
        // Only apply adjustments to non-historical agents
        const ga = geminiAdjustments;

        // Helper: get historical value at/just before simStart
        const histAt = (arr: Record<string, any>[]): number | null => {
          if (!simStart) return null;
          // Prefer exact match; else last point before simStart
          for (let i = arr.length - 1; i >= 0; i--) {
            const d = arr[i]?.date;
            if (!d) continue;
            if (d === simStart) return arr[i]?.hist ?? null;
            if (d < simStart) return arr[i]?.hist ?? null;
          }
          return null;
        };

        const histStress0 = histAt(stress);
        const histGrowth0 = histAt(growth);
        const histCrisis0 = (() => {
          if (!timeseries || !simStart) return null;
          const idx = timeseries.dates.findIndex((d) => d === simStart);
          if (idx >= 0 && timeseries.crisis_probability?.[idx] != null) {
            return timeseries.crisis_probability[idx] * 100;
          }
          // fallback: last <= simStart from windowed crisis array
          for (let i = crisis.length - 1; i >= 0; i--) {
            const d = crisis[i]?.date;
            if (!d) continue;
            if (d <= simStart) return crisis[i]?.hist ?? null;
          }
          return null;
        })();

        // Crisis prob definition: match historical (sigmoid on z-scored stress)
        const crisisFromStress = (stressVal: number | null | undefined): number | null => {
          if (stressVal == null) return null;
          const mean = timeseries?.stress_mean;
          const std = timeseries?.stress_std;
          if (mean == null || std == null || std === 0) return null;
          const z = (stressVal - mean) / std;
          // Same shape as backend timeseries: sigmoid(2*(z - regime_threshold_crisis))
          const threshold = timeseries?.regime_threshold_crisis ?? 1.5;
          const x = 2 * (z - threshold);
          const sig = x >= 0 ? 1 / (1 + Math.exp(-x)) : ((): number => {
            const ez = Math.exp(x);
            return ez / (1 + ez);
          })();
          return sig * 100;
        };

        for (const agentId of enabledAgents) {
          const agent = agentResults[agentId];
          if (!agent?.stress_path?.length) continue;
          agents.push(agentId);
          const isPrimary = agents.length === 1;
          const H = agent.stress_path.length;
          const isHist = agentId === "historical";
          const isHeuristic = agentId === "heuristic";
          const isRL = agentId === "rl";
          const skipAdj = isHist || isHeuristic || isRL;

          // Seed a point at simStart so overlays visually continue from history
          // (prevents the perceived "reset" at the boundary).
          const key0 = `agent_${agentId}`;
          if (histStress0 != null) {
            let pt = stress.find((d) => d.date === simStart);
            if (!pt) stress.push((pt = { date: simStart, isSim: true }));
            pt[key0] = histStress0;
          }
          if (histGrowth0 != null) {
            let pt = growth.find((d) => d.date === simStart);
            if (!pt) growth.push((pt = { date: simStart, isSim: true }));
            pt[key0] = histGrowth0;
          }
          if (histCrisis0 != null) {
            let pt = crisis.find((d) => d.date === simStart);
            if (!pt) crisis.push((pt = { date: simStart, isSim: true }));
            pt[key0] = histCrisis0;
          }

          for (let step = 0; step < H; step++) {
            const date = addMonthsEnd(simStart, step + 1);
            const key = `agent_${agentId}`;

            const merge = (
              arr: Record<string, any>[],
              fields: Record<string, any>,
            ) => {
              let pt = arr.find((d) => d.date === date);
              if (!pt) {
                pt = { date, isSim: true };
                arr.push(pt);
              }
              Object.assign(pt, fields);
            };

            // Stress adjustment — skip for historical, heuristic (Gemini-generated), and RL (pure MC)
            const stressAdj = (!skipAdj && ga) ? ga.stress : 0;
            const sf: Record<string, any> = {
              [key]: adj(agent.stress_path[step], stressAdj),
            };
            if (isPrimary && agent.stress_fan?.[step]) {
              const f = agent.stress_fan[step];
              const a = (v: number) => adj(v, stressAdj) ?? v;
              sf.fan_base = a(f.p5);
              sf.fan_lo = a(f.p25) - a(f.p5);
              sf.fan_iqr = a(f.p75) - a(f.p25);
              sf.fan_hi = a(f.p95) - a(f.p75);
            }
            merge(stress, sf);

            // Growth adjustment — skip for historical, heuristic, and RL
            const growthAdj = (!skipAdj && ga) ? ga.growth : 0;
            const gf: Record<string, any> = {
              [key]: adj(agent.growth_path[step], growthAdj),
            };
            if (isPrimary && agent.growth_fan?.[step]) {
              const f = agent.growth_fan[step];
              const a = (v: number) => adj(v, growthAdj) ?? v;
              gf.gfan_base = a(f.p5);
              gf.gfan_lo = a(f.p25) - a(f.p5);
              gf.gfan_iqr = a(f.p75) - a(f.p25);
              gf.gfan_hi = a(f.p95) - a(f.p75);
            }
            merge(growth, gf);

            // Crisis adjustment — skip for historical, heuristic, and RL
            const crisisAdj = (!skipAdj && ga) ? ga.crisis : 0;
            merge(crisis, {
              [key]: adj(
                // Match historical definition by deriving from stress (z-scored)
                crisisFromStress(agent.stress_path[step]),
                crisisAdj,
              ),
            });

            // ES95 adjustment — skip for historical, heuristic, and RL
            const es95Adj = (!skipAdj && ga) ? ga.es95 : 0;
            const esVal =
              agent.es95_path?.[step] != null
                ? Number(agent.es95_path[step])
                : agent.stress_fan?.[step]
                  ? agent.stress_fan[step].p95
                  : null;
            if (esVal != null) merge(es95, { [key]: adj(esVal, es95Adj) });
          }
        }
      }

      const cmp = (a: Record<string, any>, b: Record<string, any>) =>
        a.date < b.date ? -1 : a.date > b.date ? 1 : 0;
      stress.sort(cmp);
      growth.sort(cmp);
      crisis.sort(cmp);
      es95.sort(cmp);

      return {
        stressData: stress,
        growthData: growth,
        crisisData: crisis,
        es95Data: es95,
        activeAgents: agents,
      };
    }, [timeseries, windowStart, simStart, agentResults, enabledAgents, hasAgents, geminiAdjustments]);

  const stressTicks = useMemo(
    () => computeTimeTicks(stressData as { date: string }[]),
    [stressData],
  );
  const growthTicks = useMemo(
    () => computeTimeTicks(growthData as { date: string }[]),
    [growthData],
  );
  const crisisTicks = useMemo(
    () => computeTimeTicks(crisisData as { date: string }[]),
    [crisisData],
  );
  const es95Ticks = useMemo(
    () => computeTimeTicks(es95Data as { date: string }[]),
    [es95Data],
  );

  if (!timeseries && !hasAgents) {
    return (
      <div className="flex items-center justify-center h-56 text-slate-500">
        <div className="text-center space-y-2">
          <p className="text-3xl opacity-30">📈</p>
          <p className="text-sm">Refresh data to load charts</p>
        </div>
      </div>
    );
  }

  /* ── Legend ───────────────────────────────────────────────────── */
  const legend = (
    <div className="flex items-center gap-4 flex-wrap text-[10px] px-1 mb-2">
      {timeseries && (
        <span className="flex items-center gap-1.5">
          <span className="w-5 h-[2px] bg-slate-400 inline-block rounded" />
          <span className="text-slate-400">Historical</span>
        </span>
      )}
      {activeAgents.map((k) => (
        <span key={k} className="flex items-center gap-1.5">
          <span
            className="w-5 h-[2px] inline-block rounded"
            style={{ backgroundColor: AGENT_COLORS[k] || "#888" }}
          />
          <span className="text-slate-400">{AGENT_LABELS[k] || k}</span>
        </span>
      ))}
      {activeAgents.length > 0 && (
        <span className="flex items-center gap-1.5">
          <span
            className="w-4 h-2 inline-block rounded opacity-30"
            style={{
              backgroundColor: AGENT_COLORS[activeAgents[0]] || "#888",
            }}
          />
          <span className="text-slate-500">Fan (p5–p95)</span>
        </span>
      )}
    </div>
  );

  /* ── Full 2×2 grid ─────────────────────────────────────────── */
  return (
    <div className="space-y-3">
      {legend}
      <div className="grid grid-cols-2 gap-3">
        {/* Stress */}
        <ChartCard title="Stress Factor" color="text-red-400">
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={stressData} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#475569" tick={{ fontSize: 9 }} tickFormatter={fmtDateShort} ticks={stressTicks} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={fmtDateShort} />
              {dateCursor && <ReferenceLine x={dateCursor} stroke="#818cf8" strokeWidth={1.5} strokeDasharray="6 3" label={{ value: "▼", position: "top", fill: "#818cf8", fontSize: 10 }} />}
              {simStart && hasAgents && <ReferenceLine x={simStart} stroke="#6366f1" strokeWidth={1.5} strokeDasharray="4 2" />}
              <Line type="monotone" dataKey="hist" stroke="#94a3b8" strokeWidth={1.5} dot={false} name="Historical" connectNulls isAnimationActive={false} />
              <Area stackId="fan" type="monotone" dataKey="fan_base" fill="transparent" stroke="none" isAnimationActive={false} connectNulls />
              <Area stackId="fan" type="monotone" dataKey="fan_lo" fill="#ef4444" fillOpacity={0.12} stroke="none" isAnimationActive={false} connectNulls />
              <Area stackId="fan" type="monotone" dataKey="fan_iqr" fill="#f97316" fillOpacity={0.25} stroke="none" isAnimationActive={false} connectNulls />
              <Area stackId="fan" type="monotone" dataKey="fan_hi" fill="#ef4444" fillOpacity={0.12} stroke="none" isAnimationActive={false} connectNulls />
              {activeAgents.map((k) => (
                <Line key={`${k}_${animKey}`} type="monotone" dataKey={`agent_${k}`} stroke={AGENT_COLORS[k]} strokeWidth={2} strokeDasharray="6 3" dot={false} name={AGENT_LABELS[k]} isAnimationActive={true} animationDuration={1500} animationEasing="ease-in-out" connectNulls />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Growth */}
        <ChartCard title="Growth Factor" color="text-emerald-400">
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={growthData} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#475569" tick={{ fontSize: 9 }} tickFormatter={fmtDateShort} ticks={growthTicks} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={fmtDateShort} />
              {dateCursor && <ReferenceLine x={dateCursor} stroke="#818cf8" strokeWidth={1.5} strokeDasharray="6 3" label={{ value: "▼", position: "top", fill: "#818cf8", fontSize: 10 }} />}
              {simStart && hasAgents && <ReferenceLine x={simStart} stroke="#6366f1" strokeWidth={1.5} strokeDasharray="4 2" />}
              <Line type="monotone" dataKey="hist" stroke="#94a3b8" strokeWidth={1.5} dot={false} name="Historical" connectNulls isAnimationActive={false} />
              <Area stackId="fan" type="monotone" dataKey="gfan_base" fill="transparent" stroke="none" isAnimationActive={false} connectNulls />
              <Area stackId="fan" type="monotone" dataKey="gfan_lo" fill="#22c55e" fillOpacity={0.12} stroke="none" isAnimationActive={false} connectNulls />
              <Area stackId="fan" type="monotone" dataKey="gfan_iqr" fill="#4ade80" fillOpacity={0.25} stroke="none" isAnimationActive={false} connectNulls />
              <Area stackId="fan" type="monotone" dataKey="gfan_hi" fill="#22c55e" fillOpacity={0.12} stroke="none" isAnimationActive={false} connectNulls />
              {activeAgents.map((k) => (
                <Line key={`${k}_${animKey}`} type="monotone" dataKey={`agent_${k}`} stroke={AGENT_COLORS[k]} strokeWidth={2} strokeDasharray="6 3" dot={false} name={AGENT_LABELS[k]} isAnimationActive={true} animationDuration={1500} animationEasing="ease-in-out" connectNulls />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Crisis Probability */}
        <ChartCard title="Crisis Probability (%)" color="text-amber-400">
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={crisisData} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#475569" tick={{ fontSize: 9 }} tickFormatter={fmtDateShort} ticks={crisisTicks} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} domain={[0, 100]} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={fmtDateShort} formatter={(v: number) => [`${v?.toFixed(1)}%`]} />
              {dateCursor && <ReferenceLine x={dateCursor} stroke="#818cf8" strokeWidth={1.5} strokeDasharray="6 3" label={{ value: "▼", position: "top", fill: "#818cf8", fontSize: 10 }} />}
              {simStart && hasAgents && <ReferenceLine x={simStart} stroke="#6366f1" strokeWidth={1.5} strokeDasharray="4 2" />}
              <Line type="monotone" dataKey="hist" stroke="#94a3b8" strokeWidth={1.5} dot={false} name="Historical" connectNulls isAnimationActive={false} />
              {activeAgents.map((k) => (
                <Line key={`${k}_${animKey}`} type="monotone" dataKey={`agent_${k}`} stroke={AGENT_COLORS[k]} strokeWidth={2} strokeDasharray="6 3" dot={false} name={AGENT_LABELS[k]} isAnimationActive={true} animationDuration={1500} animationEasing="ease-in-out" connectNulls />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Expected Shortfall */}
        <ChartCard title="Expected Shortfall (ES95)" color="text-purple-400">
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={es95Data} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#475569" tick={{ fontSize: 9 }} tickFormatter={fmtDateShort} ticks={es95Ticks} />
              <YAxis stroke="#475569" tick={{ fontSize: 9 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={fmtDateShort} />
              {dateCursor && <ReferenceLine x={dateCursor} stroke="#818cf8" strokeWidth={1.5} strokeDasharray="6 3" label={{ value: "▼", position: "top", fill: "#818cf8", fontSize: 10 }} />}
              {simStart && hasAgents && <ReferenceLine x={simStart} stroke="#6366f1" strokeWidth={1.5} strokeDasharray="4 2" />}
              {activeAgents.map((k) => (
                <Line key={`${k}_${animKey}`} type="monotone" dataKey={`agent_${k}`} stroke={AGENT_COLORS[k]} strokeWidth={2} strokeDasharray="6 3" dot={false} name={AGENT_LABELS[k]} isAnimationActive={true} animationDuration={1500} animationEasing="ease-in-out" connectNulls />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

/* ── Sub-component ──────────────────────────────────────────────── */

function ChartCard({
  title,
  color,
  children,
}: {
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-slate-900/80 rounded-xl border border-slate-700/30 p-4">
      <h3 className={`text-[11px] font-bold uppercase tracking-wider mb-1 ${color}`}>
        {title}
      </h3>
      {children}
    </div>
  );
}
