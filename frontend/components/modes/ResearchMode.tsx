/**
 * ResearchMode — Structural diagnostics view.
 *
 * Contains everything from Analyst mode plus:
 *   - Correlation matrix heatmap
 *   - Eigenvalue spectrum
 *   - Latent state decomposition
 *   - MSI component decomposition
 *   - Fed Policy stance detail
 *
 * All structural panels use collapsible sections.
 */

import React from "react";
import MacroCards from "../shared/MacroCards";
import Charts from "../shared/Charts";
import ScenarioPanel from "../shared/ScenarioPanel";
import AdvancedAnalytics from "../shared/AdvancedAnalytics";
import type { TimeseriesData, AgentResult, GeminiAdjustments } from "../shared/Charts";

interface ResearchModeProps {
  stressScore: number;
  growthFactor: number;
  inflationGap: number;
  fedRate: number;
  regime: string;
  timeseries: TimeseriesData | null;
  selectedDate: string | null;
  agentResults: Record<string, AgentResult> | null;
  enabledAgents: string[];
  correlationMatrix: number[][];
  correlationLabels: string[];
  eigenvalues: number[];
  mu_T: number[];
  /** Date cursor for chart sync line */
  dateCursor?: string | null;
  /** Gemini AI insight text */
  aiInsight?: string | null;
  /** Gemini trajectory adjustments for chart overlays */
  geminiAdjustments?: GeminiAdjustments | null;
}

export default function ResearchMode({
  stressScore,
  growthFactor,
  inflationGap,
  fedRate,
  regime,
  timeseries,
  selectedDate,
  agentResults,
  enabledAgents,
  correlationMatrix,
  correlationLabels,
  eigenvalues,
  mu_T,
  dateCursor,
  aiInsight,
  geminiAdjustments,
}: ResearchModeProps) {
  return (
    <div className="space-y-4">
      {/* 5 Core Indicators */}
      <MacroCards
        stressScore={stressScore}
        growthFactor={growthFactor}
        inflationGap={inflationGap}
        fedRate={fedRate}
        regime={regime}
      />

      {/* Full 2×2 Chart Grid */}
      <Charts
        timeseries={timeseries}
        selectedDate={selectedDate}
        agentResults={agentResults}
        enabledAgents={enabledAgents}
        layout="full"
        dateCursor={dateCursor}
        geminiAdjustments={geminiAdjustments}
      />

      {/* Agent Comparison */}
      <ScenarioPanel agentResults={agentResults as any} />

      {/* Structural Diagnostics — collapsible */}
      <AdvancedAnalytics
        correlationMatrix={correlationMatrix}
        correlationLabels={correlationLabels}
        eigenvalues={eigenvalues}
        mu_T={mu_T}
        stressScore={stressScore}
        regime={regime}
        inflationGap={inflationGap}
        fedRate={fedRate}
        aiInsight={aiInsight}
      />
    </div>
  );
}
