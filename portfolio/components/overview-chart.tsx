"use client";

import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { CurvePoint } from "@/lib/curves";

export interface OverviewEnv {
  id: string;
  name: string;
  accent: string;
  points: CurvePoint[];
}

export interface OverviewChartProps {
  envs: OverviewEnv[];
  height?: number;
}

interface Row {
  pct: number;
  [envId: string]: number;
}

export default function OverviewChart({ envs, height = 320 }: OverviewChartProps) {
  const rows: Row[] = Array.from({ length: 101 }, (_, pct) => ({ pct }));

  const normalized = envs.map((env) => {
    const scale = env.points.reduce((max, p) => Math.max(max, Math.abs(p.value)), 0);
    return { env, scale: scale !== 0 ? scale : 1 };
  });

  for (const { env, scale } of normalized) {
    const points = [...env.points].sort((a, b) => a.gen - b.gen);
    const total = points.length > 0 ? points[points.length - 1].gen : 0;
    let peak = 0;
    let cursor = 0;
    for (let pct = 0; pct <= 100; pct++) {
      const target = (total * pct) / 100;
      while (cursor < points.length && points[cursor].gen <= target) {
        if (Math.abs(points[cursor].value) > peak) peak = Math.abs(points[cursor].value);
        cursor++;
      }
      rows[pct][env.id] = (peak / scale) * 100;
    }
  }

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid stroke="#333" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="pct"
            type="number"
            domain={[0, 100]}
            stroke="#888"
            tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
            label={{ value: "% of run length", position: "insideBottom", offset: -14, fontSize: 11, fill: "#888" }}
          />
          <YAxis
            width={48}
            domain={[0, 100]}
            stroke="#888"
            tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
            label={{ value: "% of peak best reward", position: "insideLeft", angle: -90, fontSize: 11, fill: "#888" }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{ fontFamily: "var(--font-mono)", fontSize: 12 }}
            formatter={(value, name) => [`${Number(value).toFixed(0)}% of its peak`, name]}
            labelFormatter={(label) => `${Number(label).toFixed(0)}% through the run`}
          />
          {envs.length > 1 ? (
            <Legend wrapperStyle={{ fontFamily: "var(--font-mono)", fontSize: 11 }} />
          ) : null}
          {envs.map((env) => (
            <Line
              key={env.id}
              type="monotone"
              dataKey={env.id}
              name={env.name}
              stroke={env.accent}
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}