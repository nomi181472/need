import curvesJson from "@/content/curves.json";

export type SeriesKey = "best_reward" | "mean_reward" | "median_reward" | "coverage" | "qd_score";

export interface CurvePoint {
  gen: number;
  value: number;
}

export interface CurveRun {
  name: string;
  series: Partial<Record<SeriesKey, CurvePoint[]>>;
}

interface CurvesFile {
  environments: Record<string, { runs: Array<{ name: string; series: Record<string, [number, number][]> }> }>;
}

const curves = curvesJson as unknown as CurvesFile;

function rawToPoints(raw: [number, number][]): CurvePoint[] {
  return raw.map(([gen, value]) => ({ gen, value }));
}

export function envCurveRuns(envId: string): CurveRun[] {
  const env = curves.environments[envId];
  if (!env) return [];
  return env.runs.map((run) => {
    const series: CurveRun["series"] = {};
    for (const [key, points] of Object.entries(run.series)) {
      series[key as SeriesKey] = rawToPoints(points);
    }
    return { name: run.name, series };
  });
}

export function hasCurves(envId: string): boolean {
  return curves.environments[envId]?.runs.length > 0;
}

function finalBest(run: CurveRun): number {
  const best = run.series.best_reward;
  return best && best.length > 0 ? best[best.length - 1].value : -Infinity;
}

export function bestCurveRun(envId: string): CurveRun | null {
  const runs = envCurveRuns(envId);
  if (runs.length === 0) return null;
  return runs.reduce((a, b) => (finalBest(b) > finalBest(a) ? b : a));
}

export function bestRunCoverage(envId: string): CurveRun | null {
  const runs = envCurveRuns(envId).filter((run) => (run.series.coverage?.length ?? 0) > 0);
  if (runs.length === 0) return null;
  return runs.reduce((a, b) => (finalBest(b) > finalBest(a) ? b : a));
}