import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { buildManifest } from "./generate-manifest.mjs";

let fixtureRoot;
const mk = (...parts) => path.join(fixtureRoot, ...parts);

test.before(async () => {
  fixtureRoot = await mkdtemp(path.join(tmpdir(), "manifest-test-"));

  await mkdir(mk("dataX", ".hidden"), { recursive: true });
  await writeFile(mk("dataX", ".hidden", "secret.mp4"), "x");
  await mkdir(mk("dataX", "EnvA-v5"), { recursive: true });
  await writeFile(mk("dataX", "EnvA-v5", ".dot.mp4"), "x");
  await writeFile(mk("dataX", "EnvA-v5", "notes.txt"), "x");
  await writeFile(mk("dataX", "EnvA-v5", "model1.pth"), "w1");
  await writeFile(mk("dataX", "EnvA-v5", "model2.pth"), "w2");
  await writeFile(
    mk("dataX", "EnvA-v5", "videos_reward_-15_run_15_vb82_seed_10_18040_20250516_133539_373580-episode-0.mp4"),
    "x",
  );
  await writeFile(mk("dataX", "EnvA-v5", "ensemble-06D-episode-34.mp4"), "x");
  await writeFile(mk("dataX", "EnvA-v5", "final__.mp4"), "x");

  await mkdir(mk("dataY", "EnvB-v1"), { recursive: true });
  await writeFile(mk("dataY", "EnvB-v1", "run1.mp4"), "x");
  await mkdir(mk("dataY", "EnvA-v5"), { recursive: true });
  await writeFile(mk("dataY", "EnvA-v5", "model1.pth"), "w1");

  await mkdir(mk("dataX2", "EnvA-v5", "recordings"), { recursive: true });
  await writeFile(mk("dataX2", "EnvA-v5", "recordings", "dup.mp4"), "x");
  await mkdir(mk("dataY2", "EnvA-v5", "recordings"), { recursive: true });
  await writeFile(mk("dataY2", "EnvA-v5", "recordings", "dup.mp4"), "y");
});

test.after(async () => {
  await rm(fixtureRoot, { recursive: true, force: true });
});

test("buildManifest parses reward and run metadata", async () => {
  const manifest = await buildManifest({ roots: [mk("dataX")] });
  const training = manifest.recordings.find(
    (r) => r.filename === "videos_reward_-15_run_15_vb82_seed_10_18040_20250516_133539_373580-episode-0.mp4",
  );
  assert.ok(training);
  assert.equal(training.fitness, -15);
  assert.equal(training.generation, 15);
  assert.equal(training.kind, "training");
  assert.equal(training.environment, "EnvA-v5");
  assert.equal(training.source, "dataX");
  assert.equal(
    training.path,
    "dataX/EnvA-v5/videos_reward_-15_run_15_vb82_seed_10_18040_20250516_133539_373580-episode-0.mp4",
  );
  assert.equal(training.episode, 0);
});

test("buildManifest classifies ensemble and plain episodes", async () => {
  const manifest = await buildManifest({ roots: [mk("dataX")] });
  const ensemble = manifest.recordings.find((r) => r.filename.startsWith("ensemble-06D"));
  assert.equal(ensemble.kind, "ensemble");
  assert.equal(ensemble.fitness, null);
  assert.equal(ensemble.generation, null);
  assert.equal(ensemble.episode, 34);
  const plain = manifest.recordings.find((r) => r.filename === "final__.mp4");
  assert.equal(plain.kind, "episode");
  assert.equal(plain.fitness, null);
  assert.equal(plain.episode, null);
});

test("buildManifest skips hidden files, non-mp4", async () => {
  const manifest = await buildManifest({ roots: [mk("dataX")] });
  assert.ok(!manifest.recordings.some((r) => r.path.includes("hidden")));
  assert.ok(!manifest.recordings.some((r) => r.path.includes(".dot.mp4")));
  assert.ok(!manifest.recordings.some((r) => r.path.endsWith(".pth")));
  const envA = manifest.environments.find((e) => e.id === "EnvA-v5");
  assert.equal(envA.weightCount, 2);
});

test("buildManifest keeps empty environments and aggregates ids", async () => {
  const manifest = await buildManifest({ roots: [mk("dataX"), mk("dataY")] });
  const envB = manifest.environments.find((e) => e.id === "EnvB-v1");
  assert.ok(envB);
  assert.equal(envB.recordingCount, 1);
  const envA = manifest.environments.find((e) => e.id === "EnvA-v5");
  assert.ok(envA);
  assert.equal(envA.recordingCount, 3);
  assert.equal(envA.weightCount, 3);
});

test("buildManifest keeps cross-root same relative filename distinct", async () => {
  const manifest = await buildManifest({ roots: [mk("dataX2"), mk("dataY2")] });
  const dupes = manifest.recordings.filter((r) => r.filename === "dup.mp4");
  assert.equal(dupes.length, 2);
  const paths = new Set(dupes.map((r) => r.path));
  assert.equal(paths.size, 2);
  assert.ok(paths.has("dataX2/EnvA-v5/recordings/dup.mp4"));
  assert.ok(paths.has("dataY2/EnvA-v5/recordings/dup.mp4"));
  const ids = new Set(dupes.map((r) => r.id));
  assert.equal(ids.size, 2);
  for (const r of dupes) {
    assert.equal(r.source, r.path.split("/")[0]);
    assert.ok(r.path.startsWith(`${r.source}/`));
  }
});

test("buildManifest totals bytes and stable short sha1 ids", async () => {
  const manifest = await buildManifest({ roots: [mk("dataX")] });
  const expectedBytes = manifest.recordings.reduce((sum, r) => sum + r.bytes, 0);
  assert.equal(manifest.totalBytes, expectedBytes);
  for (const r of manifest.recordings) {
    assert.match(r.id, /^[0-9a-f]{12}-/);
    assert.ok(r.path.startsWith(`${r.source}/`));
  }
  const again = await buildManifest({ roots: [mk("dataX")] });
  assert.deepEqual(again.recordings.map((r) => r.id), manifest.recordings.map((r) => r.id));
});
