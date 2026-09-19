import { createHash } from "node:crypto";
import { mkdir, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";

const PROJECT_ROOT = path.resolve(import.meta.dirname, "..");

const ROOTS = [
  path.resolve(PROJECT_ROOT, "..", "data"),
  path.resolve(PROJECT_ROOT, "..", "data_2"),
];

const REWARD_RUN_RE = /videos_reward_(-?\d+)_run_(\d+)/;
const EPISODE_RE = /-episode-(\d+)\.mp4$/;

function recordingId(rel) {
  const segments = rel.split("/");
  const short = segments.length > 1 ? `${segments[segments.length - 2]}/${segments[segments.length - 1]}` : segments[0];
  return `${createHash("sha1").update(rel).digest("hex").slice(0, 12)}-${short}`;
}

function kindFor(basename) {
  if (basename.startsWith("ensemble-")) return "ensemble";
  if (REWARD_RUN_RE.test(basename)) return "training";
  return "episode";
}

export async function buildManifest({ roots = ROOTS } = {}) {
  const recordings = [];
  const environments = [];
  const envIndex = new Map();
  let totalBytes = 0;

  const ensureEnv = (id) => {
    let env = envIndex.get(id);
    if (!env) {
      env = { id, recordingCount: 0, weightCount: 0 };
      envIndex.set(id, env);
      environments.push(env);
    }
    return env;
  };

  for (const root of roots) {
    let rootEntries;
    try {
      rootEntries = await readdir(root, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const envEntry of rootEntries) {
      if (envEntry.name.startsWith(".") || envEntry.isFile() || envEntry.isSymbolicLink()) continue;
      const envDir = path.join(root, envEntry.name);
      let envStat;
      try {
        envStat = await stat(envDir);
      } catch {
        continue;
      }
      if (!envStat.isDirectory()) continue;

      const env = ensureEnv(envEntry.name);
      const queue = [envDir];
      while (queue.length > 0) {
        const dir = queue.shift();
        let dirents;
        try {
          dirents = await readdir(dir, { withFileTypes: true });
        } catch {
          continue;
        }
        for (const dirent of dirents) {
          if (dirent.name.startsWith(".")) continue;
          const target = path.join(dir, dirent.name);
          if (dirent.isSymbolicLink()) continue;
          if (dirent.isDirectory()) {
            queue.push(target);
            continue;
          }
          if (!dirent.isFile()) continue;
          if (dirent.name.toLowerCase().endsWith(".mp4")) {
            let info;
            try {
              info = await stat(target);
            } catch {
              continue;
            }
            const source = path.basename(root);
            const rel = `${source}/${path.relative(root, target).split(path.sep).join("/")}`;
            const match = REWARD_RUN_RE.exec(dirent.name);
            const episodeMatch = EPISODE_RE.exec(dirent.name);
            const record = {
              id: recordingId(rel),
              path: rel,
              environment: envEntry.name,
              source,
              filename: dirent.name,
              fitness: match ? Number(match[1]) : null,
              generation: match ? Number(match[2]) : null,
              episode: episodeMatch ? Number(episodeMatch[1]) : null,
              kind: kindFor(dirent.name),
              bytes: info.size,
            };
            recordings.push(record);
            env.recordingCount += 1;
            totalBytes += info.size;
          } else if (dirent.name.toLowerCase().endsWith(".pth")) {
            env.weightCount += 1;
          }
        }
      }
    }
  }

  recordings.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  environments.sort((a, b) => a.id.localeCompare(b.id));
  return { recordings, environments, totalBytes };
}

const invokedDirectly =
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(import.meta.filename);

if (invokedDirectly) {
  const manifest = await buildManifest();
  const contentDir = path.resolve(import.meta.dirname, "..", "content");
  await mkdir(contentDir, { recursive: true });
  await writeFile(path.join(contentDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(
    `manifest: ${manifest.recordings.length} recordings, ${manifest.environments.length} environments, ${manifest.totalBytes} bytes`,
  );
}
