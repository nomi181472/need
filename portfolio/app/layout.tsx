import type { Metadata } from "next";
import "@fontsource/space-grotesk/400.css";
import "@fontsource/space-grotesk/500.css";
import "@fontsource/space-grotesk/600.css";
import "@fontsource/space-grotesk/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import Navigation from "@/components/navigation";
import Link from "next/link";
import { FULL_LABEL, GITHUB_URL } from "@/lib/site";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: `NEED — NeuroEvolution for Effective Decision · ${FULL_LABEL}`, template: "%s | NEED" },
  description: "NEED, a master's final-year project in neuroevolution by Noman Ali: neural policies evolved from scratch, without backpropagation, to learn to act. Explore real recordings, compare generations, and see evolutionary search in motion.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#content">Skip to content</a>
        <Navigation />
        <div id="content">{children}</div>
        <footer className="site-footer">
          <Link className="wordmark" href="/v1">e/m<span>EVOLUTION IN MOTION</span></Link>
          <p>A {FULL_LABEL.toLowerCase()} field journal of emergent behavior.</p>
          <Link href="/v1/methodology">Built on experiments, not promises. ↗</Link>
          <a className="text-link" href={GITHUB_URL} target="_blank" rel="noreferrer">Source &amp; logs on GitHub ↗</a>
        </footer>
      </body>
    </html>
  );
}
