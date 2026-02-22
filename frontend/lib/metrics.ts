/**
 * Standardized metric definitions, agent constants, and formatting utilities.
 * Used across every component to ensure consistent naming, colors, and units.
 */

/* ── Agent constants ────────────────────────────────────────────── */

export const AGENT_COLORS: Record<string, string> = {
  heuristic: "#10b981",
  rl: "#a855f7",
  historical: "#f97316",
};

export const AGENT_LABELS: Record<string, string> = {
  heuristic: "Heuristic Agent",
  rl: "RL Agent",
  historical: "Historical Fed Policy",
};

export const AGENT_ICONS: Record<string, string> = {
  heuristic: "📐",
  rl: "🧠",
  historical: "🏛",
};

export const AGENT_DESCS: Record<string, string> = {
  heuristic: "MC optimizer with objective weights",
  rl: "Trained PPO neural policy",
  historical: "Actual Fed rate decisions (pre-2026 only)",
};

/* ── Regime helpers ─────────────────────────────────────────────── */

export function regimeBg(regime: string): string {
  switch (regime) {
    case "Normal":
      return "bg-green-900/40 text-green-300";
    case "Fragile":
      return "bg-amber-900/40 text-amber-300";
    case "Crisis":
      return "bg-red-900/40 text-red-300";
    default:
      return "bg-slate-800 text-slate-300";
  }
}

export function regimeGradient(regime: string): string {
  switch (regime) {
    case "Normal":
      return "from-green-600 to-emerald-600";
    case "Fragile":
      return "from-yellow-600 to-amber-600";
    case "Crisis":
      return "from-red-600 to-rose-600";
    default:
      return "from-slate-600 to-slate-700";
  }
}

/* ── Crisis probability (same sigmoid used by backend) ──────────── */

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

const CRISIS_THRESHOLD_Z = 1.5;

export function computeCrisisProb(stressScore: number): number {
  return sigmoid(2 * (stressScore - CRISIS_THRESHOLD_Z));
}

/* ── Date formatting ────────────────────────────────────────────── */

const MONTH_NAMES = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** "2024-01-31" → "Jan 24" */
export function fmtDateShort(d: string | undefined): string {
  if (!d) return "";
  const parts = d.split("-");
  if (parts.length < 2) return d;
  return `${MONTH_NAMES[parseInt(parts[1], 10) - 1]} ${parts[0].slice(2)}`;
}

/** "2024-01-31" → "Jan 2024" */
export function fmtDateLong(d: string): string {
  const parts = d.split("-");
  if (parts.length < 2) return d;
  return `${MONTH_NAMES[parseInt(parts[1], 10) - 1]} ${parts[0]}`;
}

/** Compute end-of-month date N months after a YYYY-MM-DD string. */
export function addMonthsEnd(dateStr: string, n: number): string {
  const [y, m] = dateStr.split("-").map(Number);
  const total = y * 12 + (m - 1) + n;
  const newYear = Math.floor(total / 12);
  const newMonth = (total % 12) + 1;
  const lastDay = new Date(newYear, newMonth, 0).getDate();
  return `${newYear}-${String(newMonth).padStart(2, "0")}-${String(lastDay).padStart(2, "0")}`;
}

/** Pick evenly-spaced date ticks from a sorted data array. */
export function computeTimeTicks(
  data: { date: string }[],
  targetCount = 8,
): string[] {
  const dates = data.map((d) => d.date).filter(Boolean);
  if (dates.length <= targetCount) return dates;
  const step = Math.ceil(dates.length / (targetCount - 1));
  const ticks: string[] = [];
  for (let i = 0; i < dates.length; i += step) ticks.push(dates[i]);
  if (ticks[ticks.length - 1] !== dates[dates.length - 1]) {
    ticks.push(dates[dates.length - 1]);
  }
  return ticks;
}

/* ── Recharts tooltip style ─────────────────────────────────────── */

export const TOOLTIP_STYLE = {
  backgroundColor: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 8,
  fontSize: 11,
};
