import type { Metadata } from "next";
import Link from "next/link";
import {
  RESEARCH,
  environmentsWithDisplay,
  featuredRecording,
  recordings,
  totalBytes,
} from "@/lib/data";
import Video from "@/components/video";

export const metadata: Metadata = {
  title: "Nine experiments in learned motion",
  description:
    "Master's thesis research in neuroevolution: neural controllers for Gymnasium agents evolved from scratch, aiming to improve learning without backpropagation. Every video is a real experiment run.",
};

function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}

function formatBytes(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  if (mb >= 1) return `${mb.toFixed(0)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export default function Home() {
  const environments = environmentsWithDisplay();
  const featured = environments
    .map((environment) => ({ environment, recording: featuredRecording(environment.id) }))
    .filter((pair): pair is typeof pair & { recording: NonNullable<typeof pair.recording> } => pair.recording !== null);
  const [heroA, heroB, heroC] = featured;
  const trainingClips = recordings.filter((recording) => recording.kind === "training");
  const genesis = recordings.find((recording) => recording.environment === "CartPole-v1") ?? recordings[0];
  const archiveGB = totalBytes / (1024 * 1024 * 1024);

  return (
    <div className="index-shell">
      <main className="index-main">
        <section className="grid-fill">
          <div className="hero">
            <p className="eyebrow teal">{RESEARCH.title} / Neuroevolution</p>
            <h1 className="hero-heading">
              Morphologies in motion, <em>evolved from noise.</em>
            </h1>
            <p className="hero-sub">
              Nine simulated bodies — cheetahs, walkers, humanoids, swimmers — with one thing in
              common: no human wrote their controllers, and no gradient ever updated them. This
              is my {RESEARCH.title.toLowerCase()} research: evolving neural policies the way
              nature does, to improve learning without backpropagation.
            </p>
            <div className="hero-actions">
              <Link className="button button-primary" href="/gallery">
                Browse evidence <span className="arrow">→</span>
              </Link>
              <Link className="button" href="/methodology">
                How it works
              </Link>
              <Link className="button button-ghost" href="/compare">
                Compare two runs
              </Link>
            </div>
          </div>
        </section>

        <section className="index-content">
          <div className="stats-strip" aria-label="Project totals">
            <div className="stat-box">
              <p className="stat-label">Environments</p>
              <p className="stat-value">{environments.length}</p>
            </div>
            <div className="stat-box">
              <p className="stat-label">Recordings on tape</p>
              <p className="stat-value">{formatCount(recordings.length)}</p>
            </div>
            <div className="stat-box">
              <p className="stat-label">Training clips</p>
              <p className="stat-value">{formatCount(trainingClips.length)}</p>
            </div>
            <div className="stat-box">
              <p className="stat-label">Media archived</p>
              <p className="stat-value">
                {archiveGB >= 1 ? archiveGB.toFixed(1) : formatBytes(totalBytes)}
                {archiveGB >= 1 ? <sup>GB</sup> : null}
              </p>
            </div>
            <div className="stat-box">
              <p className="stat-label">Research days</p>
              <p className="stat-value">
                {RESEARCH.seasonDays}
                <sup>Jan–Jul</sup>
              </p>
            </div>
          </div>

          <section aria-labelledby="run-heading">
            <div className="index-title-row">
              <div>
                <h2 className="index-section-heading" id="run-heading">
                  The run
                </h2>
                <p className="index-section-lead">
                  {RESEARCH.seasonLabel}. From the first genetic loop to the last recording — one
                  aim throughout: learn to act without backpropagation.
                </p>
              </div>
              <Link className="link-arrow text-link" href="/methodology">
                <span className="arrow">→</span> Methodology
              </Link>
            </div>
            <div className="research-strip">
              <div className="research-phase">
                <p className="stat-label">2024</p>
                <p className="muted">
                  Groundwork year. The genetic loop takes shape — populations of linear
                  policies, fitness-driven selection, zero gradients anywhere.
                </p>
              </div>
              <div className="research-phase">
                <p className="stat-label">1 Jan – 31 Jul 2025 · 212 days</p>
                <p className="muted">
                  Thesis season. Operator grid sweeps on CartPole and Swimmer, then the main
                  runs — Lunar Lander, walkers, cheetahs, swimmers, and a humanoid learning to
                  stand.
                </p>
              </div>
              <div className="research-phase">
                <p className="stat-label">Archive · {RESEARCH.archiveStamp}</p>
                <p className="muted">
                  Best policies from the season rendered to tape. The recordings archived on
                  this site carry {RESEARCH.archiveStamp} stamps; the research itself ran from
                  2024 into July 2025.
                </p>
              </div>
            </div>
          </section>

          <section aria-labelledby="env-heading">
            <div className="index-title-row">
              <div>
                <h2 className="index-section-heading" id="env-heading">
                  The bodies
                </h2>
                <p className="index-section-lead">
                  Eight domains, each its own population and fitness story. Open one to see its
                  clips, weights, and the fitness archive for each generation.
                </p>
              </div>
              <Link className="link-arrow text-link" href="/gallery">
                <span className="arrow">→</span> All recordings
              </Link>
            </div>
            <div className="env-index-list">
              {environments.map((environment) => (
                <Link
                  className="env-index-entry"
                  key={environment.id}
                  href={`/evolutions/${environment.id}`}
                  style={{ "--dot-color": environment.accent } as React.CSSProperties}
                >
                  <span className="dot" aria-hidden="true" />
                  <span className="env-name">{environment.name}</span>
                  <span className="env-meta">
                    <span>{environment.recordingCount} clips</span>
                    <span>{environment.weightCount} weights</span>
                  </span>
                  <span className="env-cat">{environment.category}</span>
                </Link>
              ))}
            </div>
          </section>

          <section aria-labelledby="video-heading">
            <div className="index-title-row">
              <div>
                <h2 className="index-section-heading" id="video-heading">
                  Emergent behavior, playable
                </h2>
                <p className="index-section-lead">
                  The best policies from each environment, rendered to tape. These are the actual
                  archives served by this site, not promotional renders.
                </p>
              </div>
              <Link className="link-arrow text-link" href="/compare">
                <span className="arrow">→</span> Compare players
              </Link>
            </div>
            {featured.length > 0 ? (
              <div className="media-collection">
                {heroA ? (
                  <Link className="media-tile hero-video" href={`/evolutions/${heroA.environment.id}`}>
                    <Video recording={heroA.recording} autoPlay />
                    <span className="media-tile-caption">
                      <span className="env">{heroA.environment.name}</span>
                      <span className="kind">{heroA.recording.kind}</span>
                    </span>
                  </Link>
                ) : null}
                <div className="media-stage">
                  {heroB ? (
                    <Link className="media-tile" href={`/evolutions/${heroB.environment.id}`}>
                      <Video recording={heroB.recording} />
                      <span className="media-tile-caption">
                        <span className="env">{heroB.environment.name}</span>
                        <span className="kind">{heroB.recording.kind}</span>
                      </span>
                    </Link>
                  ) : null}
                  {heroC ? (
                    <Link className="media-tile" href={`/evolutions/${heroC.environment.id}`}>
                      <Video recording={heroC.recording} />
                      <span className="media-tile-caption">
                        <span className="env">{heroC.environment.name}</span>
                        <span className="kind">{heroC.recording.kind}</span>
                      </span>
                    </Link>
                  ) : null}
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <p className="stat-label">No recordings yet</p>
                <p className="muted">When the archive grows, the best of each environment will play here.</p>
              </div>
            )}
          </section>

          <section aria-labelledby="about-heading">
            <div className="index-title-row">
              <div>
                <h2 className="index-section-heading" id="about-heading">
                  Built on experiments, not promises
                </h2>
                <p className="index-section-lead">
                  This is a record of real runs — including the ones that barely moved. My{" "}
                  {RESEARCH.title.toLowerCase()}, carried out 2024 into July 2025: a small
                  genetic algorithm of my own, Gymnasium as the playground, and filename
                  metadata as the source of truth. Everything here was learned without
                  backpropagation.
                </p>
              </div>
              <Link className="link-arrow text-link" href="/methodology">
                <span className="arrow">→</span> Methodology &amp; caveats
              </Link>
            </div>
            <div className="panel-columns">
              <div className="panel">
                <p className="stat-label">The recipe</p>
                <p className="muted">
                  A population of linear policies, each a flat vector of network weights. Every
                  candidate rolls out in its environment; fitness is the mean episode reward over
                  repeats. Parents mix 70% fitness with 30% novelty — being different earns
                  reproduction rights alongside being good. Uniform crossover, polynomial mutation,
                  and a periodic connectivity-restructuring pass shake the network topology so
                  elites cannot sit still.
                </p>
              </div>
              <div className="panel">
                <p className="stat-label">What the numbers mean</p>
                <p className="muted">
                  Filename fitness is a mean training value recorded during evolution — not the
                  return of the clip you are watching, which was a separate recording run. Reward
                  shaping and action scaling changed between experiments, so numbers are not
                  directly comparable across all runs. Caveats live on the{" "}
                  <Link className="text-link" href="/methodology">methodology page</Link>.
                </p>
              </div>
            </div>
          </section>

          <section aria-labelledby="genesis-heading">
            <div className="panel-columns">
              <div>
                <p className="eyebrow teal">Where it began</p>
                <h2 className="index-section-heading" id="genesis-heading">
                  Original sin: CartPole.
                </h2>
                <p className="index-section-lead">
                  The smallest system in the archive is where the toolchain started — archived
                  weights, no video. It is the calibration run: if evolution cannot balance a pole
                  on a cart, it has no business walking a cheetah.
                </p>
                <p className="hero-sub">
                  Everything after that — Lunar Lander, the swimmers, the walkers, the humanoids
                  standing up — inherited the same loop: evolve, evaluate, record, archive.
                </p>
              </div>
              <div className="gallery-card">
                {genesis ? (
                  <Link className="media-tile" href={`/evolutions/${genesis.environment}`}>
                    <Video recording={genesis} />
                    <span className="media-tile-caption">
                      <span className="env">{genesis.environment}</span>
                      <span className="kind">gen {genesis.generation ?? "—"}</span>
                    </span>
                  </Link>
                ) : null}
                <p className="muted">
                  One archived recording: <code className="font-mono">{genesis ? genesis.filename : "—"}</code>
                </p>
              </div>
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}