import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "v2 — Coming soon",
  description: "Version 2 of the neuroevolution archive is in the works.",
};

export default function V2Page() {
  return (
    <main className="page-shell">
      <header className="page-intro">
        <p className="eyebrow">Next iteration</p>
        <h1>v2 — Coming soon</h1>
        <p className="muted">
          Version 2 of the archive is still in progress. Ahead: the future-work chapter of the
          project — neuromodulated plasticity, multi-agent coordination, and greener,
          GPU-free evolution.
        </p>
      </header>
      <section className="panel coming-soon" aria-label="Coming soon">
        <p className="stat-value">Coming soon</p>
        <p className="muted">v2 is being built. The full archive currently lives under v1.</p>
      </section>
    </main>
  );
}