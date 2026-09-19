"use client";

import { Line, LineChart, ResponsiveContainer } from "recharts";
import type { CurvePoint } from "@/lib/curves";

export interface RunGridProps {
  runs: Array<{ name: string; best: CurvePoint[] }>;
  accent: string;
}

function finalValue(points: CurvePoint[]): number | null {
  return points.length > 0 ? points[points.length - 1].value : null;
}

export default function RunGrid({ runs, accent }: RunGridProps) {
  return (
    <div className="run-grid">
      {runs.map((run) => {
        const last = finalValue(run.best);
        return (
          <figure className="run-card" key={run.name}>
            <figcaption className="run-card-head">
              <span className="run-card-name">{run.name}</span>
              <span className="run-card-value">
                {last !== null ? last.toLocaleString("en-US", { maximumFractionDigits: 0 }) : "—"}
              </span>
            </figcaption>
            <div className="run-card-chart" style={{ width: "100%", height: 64 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={run.best} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
                  <Line
                    dataKey="value"
                    stroke={accent}
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </figure>
        );
      })}
    </div>
  );
}