/**
 * macroStability.ts — Macro Stability Index (MSI) + Policy Stance logic.
 *
 * MSI is a composite score 0–100 derived from four normalized components:
 *   1. Inflation gap magnitude
 *   2. Financial stress level
 *   3. Crisis probability
 *   4. Growth volatility (absolute deviation)
 *
 * Each is normalized to [0, 1] via configurable saturation bounds,
 * combined with a weighted average, then inverted/scaled to 0–100
 * where 100 = maximum stability (all green).
 *
 * Policy Stance classifies actual vs. Taylor-implied rate:
 *   Restrictive   → deviation > +50 bps
 *   Neutral        → ±50 bps
 *   Accommodative  → deviation < −50 bps
 */

/* ── MSI weights (configurable) ─────────────────────────────────── */

export interface MSIWeights {
  inflation: number;
  stress: number;
  crisis: number;
  growthVol: number;
}

export const DEFAULT_MSI_WEIGHTS: MSIWeights = {
  inflation: 0.25,
  stress: 0.30,
  crisis: 0.30,
  growthVol: 0.15,
};

/* ── Saturation bounds for normalization ─────────────────────────── */

interface NormBounds {
  good: number; // below this → score = 0 (best)
  bad: number;  // above this → score = 1 (worst)
}

const BOUNDS: Record<string, NormBounds> = {
  inflation: { good: 0.005, bad: 0.04 },  // |inflation_gap|
  stress:    { good: 0.2,   bad: 2.0 },   // z-score
  crisis:    { good: 0.05,  bad: 0.6 },   // probability [0,1]
  growthVol: { good: 0.1,   bad: 1.5 },   // absolute growth deviation
};

/** Clamp x to [0, 1] via linear normalization between bounds. */
function normalize(value: number, bounds: NormBounds): number {
  const clamped = Math.min(Math.max(value, bounds.good), bounds.bad);
  return (clamped - bounds.good) / (bounds.bad - bounds.good);
}

/* ── MSI computation ────────────────────────────────────────────── */

export interface MSIResult {
  /** Composite score 0–100 (100 = most stable). */
  score: number;
  /** Per-component normalized severity [0,1]. */
  components: {
    inflation: number;
    stress: number;
    crisis: number;
    growthVol: number;
  };
  /** Color class for display. */
  color: string;
  /** Text label. */
  label: string;
}

export function computeMSI(
  inflationGap: number,
  stressScore: number,
  crisisProb: number,
  growthFactor: number,
  weights: MSIWeights = DEFAULT_MSI_WEIGHTS,
): MSIResult {
  const components = {
    inflation: normalize(Math.abs(inflationGap), BOUNDS.inflation),
    stress: normalize(stressScore, BOUNDS.stress),
    crisis: normalize(crisisProb, BOUNDS.crisis),
    growthVol: normalize(Math.abs(growthFactor), BOUNDS.growthVol),
  };

  const totalWeight =
    weights.inflation + weights.stress + weights.crisis + weights.growthVol;
  const weighted =
    (components.inflation * weights.inflation +
      components.stress * weights.stress +
      components.crisis * weights.crisis +
      components.growthVol * weights.growthVol) /
    totalWeight;

  // Invert: 0 (worst) → 100 (best)
  const score = Math.round((1 - weighted) * 100);

  let color: string;
  let label: string;
  if (score >= 75) {
    color = "text-green-400";
    label = "Stable";
  } else if (score >= 50) {
    color = "text-emerald-400";
    label = "Moderate";
  } else if (score >= 30) {
    color = "text-amber-400";
    label = "Elevated Risk";
  } else {
    color = "text-red-400";
    label = "Critical";
  }

  return { score, components, color, label };
}

/* ── Policy Stance ──────────────────────────────────────────────── */

export type StanceCategory = "Restrictive" | "Neutral" | "Accommodative";

export interface PolicyStance {
  category: StanceCategory;
  deviationBps: number;
  taylorRate: number;
  color: string;
  icon: string;
  description: string;
}

const STANCE_THRESHOLD_BPS = 50; // ±50 bps

/**
 * Classify policy stance from actual Fed rate vs. Taylor rule.
 * Taylor Rule: r* + 0.5 × inflationGap + 0.5 × outputGap
 *
 * Note: growthFactor is a latent z-score, not a standard output-gap
 * percentage.  We scale it down (×0.01) so the Taylor rate stays
 * in a realistic monetary-policy range.
 */
export function computePolicyStance(
  fedRate: number,
  inflationGap: number,
  growthFactor: number,
  neutralRate = 0.02,
): PolicyStance {
  // Scale the latent growth factor to an output-gap proxy (roughly ±2 pp)
  const scaledGrowth = growthFactor * 0.01;
  const taylorRate =
    neutralRate + 0.5 * inflationGap + 0.5 * scaledGrowth;
  const rawDeviation = Math.round((fedRate - taylorRate) * 10000);
  // Clamp to a realistic range so the UI never shows absurd numbers
  const deviationBps = Math.max(-200, Math.min(200, rawDeviation));

  let category: StanceCategory;
  let color: string;
  let icon: string;
  let description: string;

  if (deviationBps > STANCE_THRESHOLD_BPS) {
    category = "Restrictive";
    color = "text-red-400";
    icon = "🔺";
    description = `Policy is ${deviationBps} bps above Taylor rule — tighter than warranted`;
  } else if (deviationBps < -STANCE_THRESHOLD_BPS) {
    category = "Accommodative";
    color = "text-green-400";
    icon = "🔻";
    description = `Policy is ${Math.abs(deviationBps)} bps below Taylor rule — more accommodative`;
  } else {
    category = "Neutral";
    color = "text-slate-300";
    icon = "⏸";
    description = `Policy is within ±50 bps of Taylor rule`;
  }

  return { category, deviationBps, taylorRate, color, icon, description };
}

/* ── Objective presets ──────────────────────────────────────────── */

export type PresetName = "inflation-first" | "stability-first" | "balanced";

export interface ObjectivePreset {
  name: PresetName;
  label: string;
  description: string;
  alpha: number;   // stability weight
  beta: number;    // growth weight
  gamma: number;   // tail risk weight
  lambda: number;  // crisis exit weight
}

export const PRESETS: Record<PresetName, ObjectivePreset> = {
  "inflation-first": {
    name: "inflation-first",
    label: "Inflation First",
    description: "Prioritize price stability — aggressive on inflation gap",
    alpha: 0.5,
    beta: 2.5,
    gamma: 0.5,
    lambda: 0.5,
  },
  "stability-first": {
    name: "stability-first",
    label: "Financial Stability",
    description: "Prioritize financial stability — minimize stress and tail risk",
    alpha: 2.5,
    beta: 0.5,
    gamma: 2.0,
    lambda: 2.0,
  },
  balanced: {
    name: "balanced",
    label: "Balanced",
    description: "Equal priority across all policy objectives",
    alpha: 1.0,
    beta: 1.0,
    gamma: 1.0,
    lambda: 1.0,
  },
};
