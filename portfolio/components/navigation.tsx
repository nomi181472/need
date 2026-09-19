"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navigation() {
  const pathname = usePathname();
  return (
    <header className="site-header">
      <Link className="wordmark" href="/" aria-label="Evolution in Motion home">e/m<span>EVOLUTION<br />IN MOTION</span></Link>
      <nav aria-label="Main navigation">
        {[ ["/", "Overview"], ["/gallery", "The archive"], ["/compare", "Compare"], ["/methodology", "Methodology"] ].map(([href, label]) => (
          <Link key={href} href={href} className={pathname === href ? "nav-active" : ""} aria-current={pathname === href ? "page" : undefined}>{label}</Link>
        ))}
      </nav>
      <span className="header-status"><i /> MASTER&apos;S THESIS</span>
    </header>
  );
}
