import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import { environmentsWithDisplay, getEnvironmentDisplay, mediaUrl, recordings, featuredRecording } from "@/lib/data";
import FitnessChart from "@/components/fitness-chart";

interface EnvPageProps {
  params: Promise<{ env: string }>;
}

export function generateStaticParams() {
  return environmentsWithDisplay().map((environment) => ({ env: environment.id }));
}

export async function generateMetadata({ params }: EnvPageProps): Promise<Metadata> {
  const { env } = await params;
  const display = getEnvironmentDisplay(env);
  if (!display) return { title: "Unknown environment" };
  return {
    title: `${display.name} — evolution detail`,
    description: display.description,
  };
}

function formatFitness(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

export default async function EnvDetailPage({ params }: EnvPageProps) {
  const { env } = await params;
  const display = getEnvironmentDisplay(env);
  if (!display) notFound();

  const pool = recordings.filter((recording) => recording.environment === env);
  const featured = featuredRecording(env);
  const training = pool.filter((recording) => recording.kind === "training" && recording.fitness !== null && recording.generation !== null);
  const ensembles = pool.filter((recording) => recording.kind === "ensemble");
  const episodes = pool.filter((recording) => recording.kind === "episode");

  return (
    <main className="page-shell">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link className="text-link" href="/">Home</Link>
        <span aria-hidden="true">/</span>
        <Link className="text-link" href="/gallery">Archive</Link>
        <span aria-hidden="true">/</span>
        <span>{display.name}</span>
      </nav>

      <header className="page-intro">
        <p className="eyebrow">{display.category}</p>
        <h1>{display.name}</h1>
        <p className="muted">{display.description}</p>
      </header>

      <section className="detail-grid" aria-label="Archive counts">
        <div className="panel">
          <p className="stat-label">Recordings</p>
          <p className="stat-value">{pool.length}</p>
        </div>
        <div className="panel">
          <p className="stat-label">Archived weight sets</p>
          <p className="stat-value">{display.weightCount}</p>
        </div>
        <div className="panel">
          <p className="stat-label">Training clips</p>
          <p className="stat-value">{training.length}</p>
        </div>
        <div className="panel">
          <p className="stat-label">Ensemble evaluations</p>
          <p className="stat-value">{ensembles.length}</p>
        </div>
      </section>

      <section className="panel" aria-label="Training fitness over generations">
        <h2 className="section-heading">Fitness across generations</h2>
        {training.length > 0 ? (
          <>
            <FitnessChart
              points={training.map((recording) => ({
                generation: recording.generation as number,
                fitness: recording.fitness as number,
              }))}
              accent={display.accent}
            />
            <p className="muted">
              Each point is one recorded generation from one run; filename fitness is a mean training value, not the return of the clip itself. Points across runs are separate sessions pooled together — the plot is a scatter, not one continuous learning curve.
            </p>
          </>
        ) : (
          <p className="muted">No training metadata with fitness for this environment yet. Weights and clips exist without a fitness series to plot.</p>
        )}
      </section>

      <section className="panel" aria-label="Featured recording">
        <h2 className="section-heading">Featured recording</h2>
        {featured ? (
          <div className="comparison-grid">
            <div className="comparison-panel">
              <video
                src={mediaUrl(featured)}
                controls
                preload="none"
                muted
                playsInline
                className="recording-media"
                style={{ width: "100%", height: "auto", display: "block", background: "#000" }}
              />
              <div className="recording-body">
                <p className="recording-meta">{featured.filename}</p>
                <p className="recording-meta muted">
                  {featured.fitness !== null ? `filename fitness ${formatFitness(featured.fitness)}` : "filename fitness —"} · episode {featured.episode ?? "—"} · generation {featured.generation ?? "—"}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <p className="muted">Nothing to feature yet. This environment has archived weight sets but no video recordings.</p>
        )}
      </section>

      <section className="panel" aria-label="Clip inventory">
        <h2 className="section-heading">Clips by kind</h2>
        <div className="recording-grid">
          {[
            { label: "training", items: training },
            { label: "ensemble", items: ensembles },
            { label: "episode", items: episodes },
          ].map(({ label, items }) => (
            <article className="recording-card" key={label}>
              <div className="recording-body">
                <p className="recording-meta">{label}</p>
                <p className="stat-value">{items.length}</p>
                {items.length === 0 ? <p className="muted">None archived for this environment.</p> : null}
              </div>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
