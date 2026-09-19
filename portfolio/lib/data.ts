import manifest from "@/content/manifest.json";

export interface Recording {
  id: string;
  path: string;
  environment: string;
  source: string;
  filename: string;
  fitness: number | null;
  generation: number | null;
  episode: number | null;
  kind: "training" | "ensemble" | "episode";
  bytes: number;
}

export interface EnvironmentEntry {
  id: string;
  recordingCount: number;
  weightCount: number;
}

export interface EnvironmentDisplay extends EnvironmentEntry {
  name: string;
  category: "Locomotion" | "Classic control" | "Robotics";
  description: string;
  accent: string;
}

export const recordings: Recording[] = manifest.recordings as Recording[];
export const environments: EnvironmentEntry[] = manifest.environments;
export const totalBytes: number = manifest.totalBytes;

export const RESEARCH = {
  title: "Master's thesis",
  aim: "Learn to act without backpropagation",
  seasonLabel: "2024 → July 2025",
  startYear: 2024,
  seasonStart: "1 Jan 2025",
  seasonEnd: "31 Jul 2025",
  seasonDays: 212,
  archiveStamp: "May 2025",
} as const;

export function seasonDayCount(): number {
  return RESEARCH.seasonDays;
}

export function mediaUrl(recording: Recording): string {
  return `/api/media/${recording.path.split("/").map(encodeURIComponent).join("/")}`;
}

export function featuredRecording(environment: string): Recording | null {
  const pool = recordings.filter((r) => r.environment === environment);
  if (pool.length === 0) return null;
  const ranked = pool.filter((r): r is Recording & { fitness: number } => typeof r.fitness === "number");
  if (ranked.length > 0) {
    return ranked.reduce((best, r) => (r.fitness > best.fitness ? r : best));
  }
  return pool[0];
}

const ENVIRONMENT_DISPLAY: Record<string, Pick<EnvironmentDisplay, "name" | "category" | "description" | "accent">> = {
  "HumanoidStandup-v5": {
    name: "Humanoid Standup",
    category: "Locomotion",
    description:
      "From the ground up. Evolving neural controllers for a humanoid tasked with standing.",
    accent: "#f59e0b",
  },
  "HalfCheetah-v5": {
    name: "HalfCheetah",
    category: "Locomotion",
    description: "Finding a rhythm. A simulated cheetah explores the mechanics of forward motion.",
    accent: "#38bdf8",
  },
  "Walker2d-v5": {
    name: "Walker2d",
    category: "Locomotion",
    description: "Bipedal walker discovering stable locomotion through fitness-driven selection.",
    accent: "#a78bfa",
  },
  "Ant-v5": {
    name: "Ant",
    category: "Locomotion",
    description: "Four-legged ant robot evolving coordinated contact-rich locomotion policies.",
    accent: "#34d399",
  },
  "Humanoid-v5": {
    name: "Humanoid",
    category: "Locomotion",
    description: "The balancing act. Neural control meets the complexity of a full humanoid body.",
    accent: "#fb7185",
  },
  "LunarLander-v3": {
    name: "Lunar Lander",
    category: "Classic control",
    description: "Thruster control under gravity; evolution tunes landing policies from sparse episodes.",
    accent: "#facc15",
  },
  "CartPole-v1": {
    name: "CartPole",
    category: "Classic control",
    description: "Small system, delicate balance. Archived neural controller weights for the classic inverted pendulum.",
    accent: "#4ade80",
  },
  "Swimmer-v5": {
    name: "Swimmer",
    category: "Robotics",
    description: "Three-link fluid swimmer; weights-only runs targeting forward velocity.",
    accent: "#22d3ee",
  },
};

const ENVIRONMENT_ORDER = [
  "HumanoidStandup-v5",
  "HalfCheetah-v5",
  "Walker2d-v5",
  "Ant-v5",
  "Humanoid-v5",
  "LunarLander-v3",
  "CartPole-v1",
  "Swimmer-v5",
];

export function environmentsWithDisplay(): EnvironmentDisplay[] {
  return ENVIRONMENT_ORDER.map((id) => {
    const entry = manifest.environments.find((e) => e.id === id);
    const display = ENVIRONMENT_DISPLAY[id];
    if (!entry || !display) return null;
    return { ...entry, ...display };
  }).filter((e): e is EnvironmentDisplay => e !== null);
}

export function getEnvironmentDisplay(id: string): EnvironmentDisplay | undefined {
  const entry = manifest.environments.find((e) => e.id === id);
  const display = ENVIRONMENT_DISPLAY[id];
  if (!entry || !display) return undefined;
  return { ...entry, ...display };
}
