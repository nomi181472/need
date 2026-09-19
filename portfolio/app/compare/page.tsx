import Comparison from "@/components/comparison";

export const metadata = {
  title: "Compare recordings",
  description: "Pick two recordings from the same or different environments and watch them side by side.",
};

export default function ComparePage() {
  return (
    <main className="page-shell">
      <header className="page-intro">
        <p className="eyebrow">Side by side / Comparison</p>
        <h1>Two recordings, one frame of reference.</h1>
        <p className="muted">Independent players. Different episode lengths are expected; there is no frame lock. Use the linked controls and judge the gait, not the wall clock.</p>
      </header>
      <Comparison />
    </main>
  );
}
