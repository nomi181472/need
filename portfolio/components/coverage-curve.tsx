"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { CurvePoint } from "@/lib/curves";

export interface CoverageCurveProps {
  name: string;
  points: CurvePoint[];
  accent: string;
  height?: number;
}

export default function CoverageCurve({ name, points, accent, height = 200 }: CoverageCurveProps) {
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
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
            width={48}
            domain={[0, 100]}
            stroke="#888"
            tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
            label={{ value: "%", position: "insideLeft", offset: 0, fontSize: 11, fill: "#888" }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{ fontFamily: "var(--font-mono)", fontSize: 12 }}
            formatter={(value) => [`${Number(value).toFixed(1)}% of map`, name]}
            labelFormatter={(label) => `generation ${Number(label).toLocaleString("en-US")}`}
          />
          <Line dataKey="value" name={name} stroke={accent} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}