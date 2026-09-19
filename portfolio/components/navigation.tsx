"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { GITHUB_URL, MASTER_LABEL } from "@/lib/site";

const ROOTS = [
  { href: "/v1", label: "v1" },
  { href: "/v2", label: "v2" },
];

const V1_LINKS = [
  { href: "/v1", label: "Overview" },
  { href: "/v1/gallery", label: "The archive" },
  { href: "/v1/compare", label: "Compare" },
  { href: "/v1/methodology", label: "Methodology" },
];

export default function Navigation() {
  const pathname = usePathname();
  const inV1 = pathname === "/v1" || pathname.startsWith("/v1/");
  return (
    <header className="site-header">
      <div className="header-top">
        <Link className="wordmark" href="/v1" aria-label="Evolution in Motion home">e/m<span>EVOLUTION<br />IN MOTION</span></Link>
        <nav className="root-nav" aria-label="Project versions">
          {ROOTS.map(({ href, label }) => {
            const active = pathname === href || (href === "/v1" && inV1);
            return (
              <Link key={href} href={href} className={active ? "nav-active" : ""} aria-current={active ? "page" : undefined}>{label}</Link>
            );
          })}
        </nav>
        <a className="header-status" href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub ↗</a>
        <span className="header-status"><i /> {MASTER_LABEL}</span>
      </div>
      {inV1 ? (
        <nav className="sub-nav" aria-label="Version 1 navigation">
          {V1_LINKS.map(({ href, label }) => {
            const active = pathname === href || (href === "/v1" && pathname.startsWith("/v1/evolutions/"));
            return (
              <Link key={href} href={href} className={active ? "nav-active" : ""} aria-current={active ? "page" : undefined}>{label}</Link>
            );
          })}
        </nav>
      ) : null}
    </header>
  );
}