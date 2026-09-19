import Link from "next/link";
import { RESEARCH, environmentsWithDisplay, totalBytes } from "@/lib/data";
import { GITHUB_URL } from "@/lib/site";

export const metadata = {
  title: "Methodology",
  description: "How the NEED framework — a master's final-year project — evolved neural controllers without backpropagation, and what the numbers do and do not mean.",
};

const PIPELINE = [
  {
    step: "Population",
    body: "A population of neural policies is initialised per environment. Despite its name, LinearPolicy uses tanh hidden activations and tanh outputs for continuous actions. Its genome comprises weights and optional biases, which can be flattened to a vector.",
  },
  {
    step: "Evaluation",
    body: "Every candidate is rolled out in a Gymnasium environment; fitness is the mean episode reward across repeats. Episode length ramps through curriculum learning, and auxiliary rewards give directional signal without a single gradient step.",
  },
  {
    step: "Selection",
    body: "The heart of NEED is hybrid selection: novelty plus a greedy, median-gated look at past fitness, so being different earns reproduction rights alongside being good. Other experiment scripts expose different operators and parameters, so this is not a claim about every archived run.",
  },
  {
    step: "Variation",
    body: "Uniform crossover and polynomial mutation vary weights and biases. Each generation, elite clones undergo connectivity restructuring — neuron relocation, pruning, and module preservation — with safeguards so the network backbone is not lost.",
  },
  {
    step: "Sweeping",
    body: "Before the main runs, an operator grid searched mutation types against crossover schemes on CartPole. The winning pairing — polynomial mutation with uniform crossover — informed the training loop used for the environments archived on this site.",
  },
  {
    step: "Archive",
    body: "Best-of-generation clips and ensembles are rendered to MP4, and the manifest on this site indexes them by environment, source, and filename metadata.",
  },
];

const CAVEATS = [
  {
    title: "Filename fitness is a mean training value",
    body: "For training clips named with reward and run fields, the current recorder encodes int(sol.rewards[-1]): the latest mean evaluation reward, truncated toward zero. That fitness is not the return of the separate video rollout. Episode and ensemble filenames do not necessarily include fitness.",
  },
  {
    title: "Training and video rollouts can differ",
    body: "The environment manager includes environment-specific reward shaping. Remote training evaluation multiplies actions by config.action_bound, while VideoEvaluator passes actions without that multiplier. Archived configurations are not fully captured by the manifest, so fitness values should not be treated as directly comparable across runs or environments.",
  },
  {
    title: "Episode labels depend on the caller",
    body: "The current_episode prefix uses a caller-supplied value: gym_ga_v1.py passes current_generation to RemoteEvaluatorGym, so that prefix can be a generation. The separate -episode-N suffix is the Gymnasium recording episode index. The manifest parses that suffix as episode and only parses generation from reward/run filenames; it does not infer generation from current_episode.",
  },
  {
    title: "Dates reflect the archive, not all of the research",
    body: `This is master's final-year project research aiming to improve learning without backpropagation, presented as the NEED framework. The research ran ${RESEARCH.startYear} into ${RESEARCH.seasonEnd} (${RESEARCH.seasonDays} days counted ${RESEARCH.seasonStart} – ${RESEARCH.seasonEnd}). The recordings archived here carry ${RESEARCH.archiveStamp} filename stamps; earlier sweep and log evidence is not fully represented on this site.`,
  },
  {
    title: "Archives are not one continuous curve",
    body: "The archive mixes sessions and configurations. Charts pool training clips in a scatter plot of filename generation against filename fitness, without session grouping or per-session traces. Multiple clips can share coordinates and overlap; these points are not a continuous learning curve.",
  },
];

export default function MethodologyPage() {
  return (
    <main className="page-shell">
      <header className="page-intro">
        <p className="eyebrow">How it works / Methodology</p>
        <h1>From genomes to video.</h1>
        <p className="muted">
          This is the method behind my master&apos;s FYP, the NEED framework: evolving neural
          controllers for Gymnasium agents, aim — improve learning without backpropagation.
          Research ran {RESEARCH.seasonLabel}, {RESEARCH.seasonDays} days counted from{" "}
          {RESEARCH.seasonStart} to {RESEARCH.seasonEnd}. Evolution optimizes policy
          weights, and the best candidates are periodically recorded. The overview below
          describes the current Python sources, not a verified configuration history for every
          clip. The manifest indexes {environmentsWithDisplay().length} environments and{" "}
          {totalBytes.toLocaleString("en-US")} bytes of media.
        </p>
      </header>

      <section aria-label="Pipeline">
        <h2 className="section-heading">Pipeline</h2>
        <ol className="pipeline">
          {PIPELINE.map(({ step, body }) => (
            <li className="pipeline-step" key={step}>
              <p className="eyebrow">{step}</p>
              <p className="muted">{body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section aria-label="Caveats">
        <h2 className="section-heading">Caveats</h2>
        <div className="limitations">
          {CAVEATS.map(({ title, body }) => (
            <article className="panel" key={title}>
              <p className="eyebrow">{title}</p>
              <p className="muted">{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section aria-label="Thesis">
        <h2 className="section-heading">The project</h2>
        <p className="muted">
          A master&apos;s final year project (FYP) written by <strong>{RESEARCH.author}</strong>,
          supervised by {RESEARCH.supervisor}, for the {RESEARCH.degree}. Defended{" "}
          {RESEARCH.seasonEnd} — the front matter, framework, sources, and tfevents logs all live
          with this project under <code className="font-mono">v1/</code> and on{" "}
          <a className="text-link" href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub ↗</a>.
        </p>
      </section>

      <p className="muted">
        Numbers on this site are read directly from filenames and the manifest. No statistics are invented. <Link className="text-link" href="/v1/gallery">Browse the archive</Link> or <Link className="text-link" href="/v1/compare">compare two recordings</Link>.
      </p>
    </main>
  );
}
