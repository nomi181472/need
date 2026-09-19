"use client";

import { Scatter, ScatterChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ZAxis } from "recharts";

export interface FitnessPoint {
  generation: number;
  fitness: number;
}

export default function FitnessChart({ points, accent }: { points: FitnessPoint[]; accent: string }) {
  const height = 260;
  return (
    <figure style={{ margin: 0 }}>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
            <XAxis
              type="number"
              dataKey="generation"
              name="generation"
              stroke="#888"
              tick={{ fontSize: 11, fontFamily: "monospace" }}
              label={{ value: "generation", position: "insideBottom", offset: -14, fontSize: 11, fill: "#888" }}
            />
            <YAxis
              type="number"
              dataKey="fitness"
              name="filename fitness"
              width={72}
              stroke="#888"
              tick={{ fontSize: 11, fontFamily: "monospace" }}
            />
            <ZAxis range={[42, 42]} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{ fontFamily: "monospace", fontSize: 12 }}
              formatter={(value) => String(value)}
            />
            <Scatter data={points} fill={accent} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="muted" style={{ marginTop: 8, fontFamily: "monospace", fontSize: 12 }}>
        Each dot represents a training clip: x = filename generation, y = filename fitness (integer-truncated mean training reward). This scatter plot pools recordings without session grouping or per-session traces; dots may overlap and do not form a continuous learning curve.
      </figcaption>
    </figure>
  );
}
