import type { Metadata } from "next";
import Link from "next/link";
import Archive from "@/components/archive";
import { recordings } from "@/lib/data";

export const metadata: Metadata = {
  title: "Recording archive",
  description: "Explore archived neuroevolution recordings by environment, source, and training metadata.",
};

export default function GalleryPage() {
  return (
    <main className="page-shell">
      <header className="page-intro">
        <p className="eyebrow">Field notes / Recording archive</p>
        <h1>Evolution, on record.</h1>
        <p className="muted">{recordings.length} recordings. Different bodies, different experiments. Inspect the movement, not just the number.</p>
        <p className="muted">Filename fitness is a mean training value, not the return of the recorded episode. <Link className="text-link" href="/methodology">Read the methodology</Link></p>
      </header>
      <Archive />
    </main>
  );
}
