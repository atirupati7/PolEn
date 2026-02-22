import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

interface CorrelationHeatmapProps {
  matrix: number[][];
  labels: string[];
}

export default function CorrelationHeatmap({ matrix, labels }: CorrelationHeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !matrix || matrix.length === 0) return;

    const n = labels.length;
    const margin = { top: 30, right: 10, bottom: 10, left: 50 };
    const cellSize = 36;
    const width = cellSize * n + margin.left + margin.right;
    const height = cellSize * n + margin.top + margin.bottom;

    // Clear previous
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Color scale: blue (-1) -> white (0) -> red (+1)
    const colorScale = d3
      .scaleSequential(d3.interpolateRdBu)
      .domain([1, -1]); // Reversed for RdBu

    // Tooltip div
    const tooltipId = "corr-tooltip";
    let existing = d3.select<HTMLDivElement, unknown>(`#${tooltipId}`);
    if (existing.empty()) {
      existing = d3
        .select<HTMLBodyElement, unknown>("body")
        .append<HTMLDivElement>("div")
        .attr("id", tooltipId)
        .style("position", "absolute")
        .style("pointer-events", "none")
        .style("background", "#1e293b")
        .style("border", "1px solid #475569")
        .style("border-radius", "6px")
        .style("padding", "6px 10px")
        .style("font-size", "11px")
        .style("color", "#e2e8f0")
        .style("opacity", "0")
        .style("z-index", "9999");
    }
    const tooltip = existing;

    // Draw cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const val = matrix[i][j];
        g.append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize - 1)
          .attr("height", cellSize - 1)
          .attr("rx", 3)
          .attr("fill", colorScale(val))
          .attr("stroke", "#1e293b")
          .attr("stroke-width", 1)
          .style("cursor", "pointer")
          .on("mouseover", (event: MouseEvent) => {
            tooltip
              .style("opacity", "1")
              .html(
                `<strong>${labels[i]} × ${labels[j]}</strong><br/>ρ = ${val.toFixed(3)}`
              )
              .style("left", event.pageX + 12 + "px")
              .style("top", event.pageY - 10 + "px");
          })
          .on("mousemove", (event: MouseEvent) => {
            tooltip
              .style("left", event.pageX + 12 + "px")
              .style("top", event.pageY - 10 + "px");
          })
          .on("mouseout", () => {
            tooltip.style("opacity", "0");
          });

        // Value text for larger cells
        if (i !== j) {
          g.append("text")
            .attr("x", j * cellSize + cellSize / 2)
            .attr("y", i * cellSize + cellSize / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .attr("font-size", "8px")
            .attr("fill", Math.abs(val) > 0.5 ? "#fff" : "#94a3b8")
            .text(val.toFixed(2));
        }
      }
    }

    // Row labels
    g.selectAll(".row-label")
      .data(labels)
      .enter()
      .append("text")
      .attr("x", -4)
      .attr("y", (d: string, i: number) => i * cellSize + cellSize / 2)
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "central")
      .attr("font-size", "9px")
      .attr("fill", "#94a3b8")
      .text((d: string) => d);

    // Column labels
    g.selectAll(".col-label")
      .data(labels)
      .enter()
      .append("text")
      .attr("x", (d: string, i: number) => i * cellSize + cellSize / 2)
      .attr("y", -8)
      .attr("text-anchor", "middle")
      .attr("font-size", "9px")
      .attr("fill", "#94a3b8")
      .text((d: string) => d);

    return () => {
      tooltip.style("opacity", "0");
    };
  }, [matrix, labels]);

  return (
    <div>
      <h4 className="text-xs font-bold text-slate-400 uppercase mb-2">Correlation Matrix</h4>
      <div className="overflow-x-auto">
        <svg ref={svgRef} />
      </div>
    </div>
  );
}
