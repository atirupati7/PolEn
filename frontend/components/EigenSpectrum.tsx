import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface EigenSpectrumProps {
  eigenvalues: number[];
  labels: string[];
}

export default function EigenSpectrum({ eigenvalues, labels }: EigenSpectrumProps) {
  const data = eigenvalues.map((val, i) => ({
    name: `λ${i + 1}`,
    value: val,
    pct: (val / eigenvalues.reduce((a, b) => a + b, 0)) * 100,
  }));

  const colors = [
    "#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#8b5cf6", "#ec4899",
    "#06b6d4", "#84cc16",
  ];

  return (
    <div>
      <h4 className="text-xs font-bold text-slate-400 uppercase mb-2">Eigenvalue Spectrum</h4>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 9 }} />
          <YAxis stroke="#64748b" tick={{ fontSize: 9 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #475569",
              borderRadius: 8,
              fontSize: 11,
            }}
            formatter={(value: number, name: string, props: Record<string, any>) => [
              `${value.toFixed(3)} (${props.payload.pct.toFixed(1)}%)`,
              name,
            ]}
          />
          <Bar dataKey="value" name="Eigenvalue" radius={[3, 3, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} fillOpacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {/* Concentration ratio */}
      <div className="mt-2 text-xs text-slate-400 text-center">
        λ₁ concentration:{" "}
        <span className="font-mono font-bold text-slate-200">
          {eigenvalues.length > 0
            ? ((eigenvalues[0] / eigenvalues.reduce((a, b) => a + b, 0)) * 100).toFixed(1)
            : 0}
          %
        </span>
      </div>
    </div>
  );
}
