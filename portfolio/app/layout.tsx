import type { Metadata } from "next";
import "@fontsource/space-grotesk/400.css";
import "@fontsource/space-grotesk/500.css";
import "@fontsource/space-grotesk/600.css";
import "@fontsource/space-grotesk/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import Navigation from "@/components/navigation";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Evolution in Motion — A Neuroevolution Master's Thesis", template: "%s | Evolution in Motion" },
  description: "Master's thesis research in neuroevolution: neural controllers evolved from scratch to improve learning without backpropagation. Explore real recordings, compare generations, and see evolutionary search in motion.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#content">Skip to content</a>
        <Navigation />
        <div id="content">{children}</div>
        <footer className="site-footer">
          <Link className="wordmark" href="/">e/m<span>EVOLUTION IN MOTION</span></Link>
          <p>A master&apos;s-thesis field journal of emergent behavior.</p>
          <Link href="/methodology">Built on experiments, not promises. ↗</Link>
        </footer>
      </body>
    </html>
  );
}
