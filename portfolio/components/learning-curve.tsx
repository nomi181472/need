"use client";

import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { CurvePoint } from "@/lib/curves";

export interface LearningCurveProps {
  name: string;
  series: {
    best_reward?: CurvePoint[];
    mean_reward?: CurvePoint[];
    median_reward?: CurvePoint[];
  };
  accent: string;
  height?: number;
}

function compactValue(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return value.toFixed(0);
}

function mergeSeries(series: LearningCurveProps["series"]): Array<{
  gen: number;
  best?: number;
  mean?: number;
  median?: number;
}> {
  const gens = new Set<number>();
  for (const key of ["best_reward", "mean_reward", "median_reward"] as const) {
    for (const point of series[key] ?? []) gens.add(point.gen);
  }
  return [...gens]
    .sort((a, b) => a - b)
    .map((gen) => {
      const row: { gen: number; best?: number; mean?: number; median?: number } = { gen };
      for (const key of ["best_reward", "mean_reward", "median_reward"] as const) {
        const point = series[key]?.find((p) => p.gen === gen);
        if (point) row[key === "best_reward" ? "best" : key === "mean_reward" ? "mean" : "median"] = point.value;
      }
      return row;
    });
}

export default function LearningCurve({ series, accent, height = 240 }: LearningCurveProps) {
  const data = mergeSeries(series);
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid stroke="#333" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="gen"
            type="number"
            name="generation"
            stroke="#888"
            tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
            label={{ value: "generation", position: "insideBottom", offset: -14, fontSize: 11, fill: "#888" }}
          />
          <YAxis
            width={64}
            stroke="#888"
            tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
            tickFormatter={compactValue}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{ fontFamily: "var(--font-mono)", fontSize: 12 }}
            formatter={(value, key) => [`${Number(value).toLocaleString("en-US", { maximumFractionDigits: 1 })}`, String(key)]}
            labelFormatter={(label) => `generation ${Number(label).toLocaleString("en-US")}`}
          />
          {series.mean_reward ? (
            <Line dataKey="mean" name="mean reward" stroke={accent} strokeWidth={1.4} strokeOpacity={0.45} dot={false} isAnimationActive={false} />
          ) : null}
          {series.median_reward ? (
            <Line dataKey="median" name="median reward" stroke="#7a7a7a" strokeWidth={1.4} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          ) : null}
          {series.best_reward ? (
            <Line dataKey="best" name="best reward" stroke={accent} strokeWidth={2} dot={false} isAnimationActive={false} />
          ) : null}
          {(series.best_reward || series.mean_reward || series.median_reward) && data.length > 1 && series.mean_reward ? (
            <Legend wrapperStyle={{ fontFamily: "var(--font-mono)", fontSize: 11 }} />
          ) : null}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}