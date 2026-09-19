"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { environmentsWithDisplay, mediaUrl, recordings } from "@/lib/data";
import type { Recording } from "@/lib/data";

const PAGE_SIZE = 12;

type KindFilter = "all" | "training" | "ensemble" | "episode";
type SourceFilter = "all" | "data" | "data_2";
type SortKey = "filename" | "fitness" | "generation";

function LazyGridVideo({ recording }: { recording: Recording }) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper || inView) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: "300px" },
    );
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, [inView]);

  return (
    <div className="recording-media" ref={wrapperRef}>
      <video
        src={inView ? mediaUrl(recording) : undefined}
        muted
        preload={inView ? "metadata" : "none"}
        playsInline
        tabIndex={-1}
        style={{ width: "100%", height: "100%", objectFit: "cover", display: "block", background: "#000" }}
      />
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatFitness(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function DialogVideo({ recording }: { recording: Recording }) {
  const [failed, setFailed] = useState(false);

  const [lastRecording, setLastRecording] = useState(recording);
  if (lastRecording !== recording) {
    setLastRecording(recording);
    setFailed(false);
  }

  if (failed) {
    return (
      <div className="empty-state">
        <p className="stat-label">Playback failed</p>
        <p className="muted">The browser could not play this recording. It may still exist on the server; try again later.</p>
        <a className="text-link" href={mediaUrl(recording)} target="_blank" rel="noreferrer">Download recording</a>
      </div>
    );
  }

  return (
    <video
      className="dialog-video"
      src={mediaUrl(recording)}
      controls
      muted
      preload="none"
      playsInline
      onError={() => setFailed(true)}
      style={{ width: "100%", height: "auto", display: "block", background: "#000" }}
    />
  );
}

export default function Archive() {
  const environments = environmentsWithDisplay();
  const [env, setEnv] = useState<string>("all");
  const [kind, setKind] = useState<KindFilter>("all");
  const [source, setSource] = useState<SourceFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("filename");
  const [query, setQuery] = useState("");
  const [visible, setVisible] = useState(PAGE_SIZE);
  const [openRecording, setOpenRecording] = useState<Recording | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = recordings.filter((recording) => {
      if (env !== "all" && recording.environment !== env) return false;
      if (kind !== "all" && recording.kind !== kind) return false;
      if (source !== "all" && recording.source !== source) return false;
      if (q && !recording.filename.toLowerCase().includes(q)) return false;
      return true;
    });
    list.sort((a, b) => {
      if (sortKey === "fitness") {
        const fa = a.fitness;
        const fb = b.fitness;
        if (fa === null && fb === null) return a.filename.localeCompare(b.filename);
        if (fa === null) return 1;
        if (fb === null) return -1;
        return fb - fa;
      }
      if (sortKey === "generation") {
        const ga = a.generation;
        const gb = b.generation;
        if (ga === null && gb === null) return a.filename.localeCompare(b.filename);
        if (ga === null) return 1;
        if (gb === null) return -1;
        return gb - ga;
      }
      return a.filename.localeCompare(b.filename);
    });
    return list;
  }, [env, kind, source, sortKey, query]);

  const resetKey = `${env}|${kind}|${source}|${sortKey}|${query}`;
  const [lastReset, setLastReset] = useState(resetKey);
  if (lastReset !== resetKey) {
    setLastReset(resetKey);
    setVisible(PAGE_SIZE);
  }

  const page = filtered.slice(0, visible);

  useEffect(() => {
    if (!openRecording) return;
    const dialog = document.getElementById("archive-dialog") as HTMLDialogElement | null;
    if (!dialog || dialog.open) return;
    const opener = document.activeElement as HTMLElement | null;
    dialog.showModal();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") dialog.close();
    };
    const handleClose = () => {
      opener?.focus();
      setOpenRecording(null);
    };
    dialog.addEventListener("keydown", handleKey);
    dialog.addEventListener("close", handleClose);
    return () => {
      dialog.removeEventListener("keydown", handleKey);
      dialog.removeEventListener("close", handleClose);
    };
  }, [openRecording]);

  return (
    <section aria-label="Recording archive">
      <div className="archive-toolbar">
        <div className="filter-tabs" role="group" aria-label="Environment filter">
          <button type="button" className={env === "all" ? "active" : ""} aria-pressed={env === "all"} onClick={() => setEnv("all")}>All</button>
          {environments.map((environment) => (
            <button
              type="button"
              key={environment.id}
              className={env === environment.id ? "active" : ""}
              aria-pressed={env === environment.id}
              onClick={() => setEnv(environment.id)}
            >
              {environment.name}
            </button>
          ))}
        </div>

        <div className="filter-tabs" role="group" aria-label="Kind filter">
          {(["all", "training", "ensemble", "episode"] as KindFilter[]).map((value) => (
            <button
              type="button"
              key={value}
              className={kind === value ? "active" : ""}
              aria-pressed={kind === value}
              onClick={() => setKind(value)}
            >
              {value}
            </button>
          ))}
        </div>

        <div className="filter-tabs" role="group" aria-label="Source filter">
          {(["all", "data", "data_2"] as SourceFilter[]).map((value) => (
            <button
              type="button"
              key={value}
              className={source === value ? "active" : ""}
              aria-pressed={source === value}
              onClick={() => setSource(value)}
            >
              {value}
            </button>
          ))}
        </div>

        <label className="muted" htmlFor="archive-search">Filename</label>
        <input
          id="archive-search"
          className="search-input"
          type="search"
          placeholder="Search filenames"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />

        <label className="muted" htmlFor="archive-sort">Sort</label>
        <select
          id="archive-sort"
          className="select-input"
          value={sortKey}
          onChange={(event) => setSortKey(event.target.value as SortKey)}
        >
          <option value="filename">Filename</option>
          <option value="fitness">Fitness, best first</option>
          <option value="generation">Generation, latest first</option>
        </select>
      </div>

      {page.length === 0 ? (
        <div className="empty-state">
          <p className="stat-label">No recordings match</p>
          <p className="muted">Loosen the filters or clear the search to see the full archive.</p>
        </div>
      ) : (
        <div className="recording-grid">
          {page.map((recording) => (
            <button
              type="button"
              className="recording-card"
              key={recording.id}
              onClick={() => setOpenRecording(recording)}
            >
              <LazyGridVideo recording={recording} />
              <div className="recording-body">
                <p className="recording-meta">{recording.filename}</p>
                <div className="recording-meta">
                  <span className="badge">{recording.kind}</span>
                  <span className="badge">{recording.source}</span>
                  {recording.fitness !== null ? <span className="badge">fitness {formatFitness(recording.fitness)}</span> : null}
                  {recording.generation !== null ? <span className="badge">gen {recording.generation}</span> : null}
                </div>
                <p className="recording-meta muted">{formatBytes(recording.bytes)}</p>
              </div>
            </button>
          ))}
        </div>
      )}

      {visible < filtered.length ? (
        <button type="button" className="button button-primary" onClick={() => setVisible((value) => value + PAGE_SIZE)}>
          Load {Math.min(PAGE_SIZE, filtered.length - visible)} more of {filtered.length}
        </button>
      ) : null}

      <dialog className="dialog" id="archive-dialog" aria-label="Recording detail">
        {openRecording ? (
          <>
            <div className="dialog-header">
              <p className="recording-meta">{openRecording.filename}</p>
              <button type="button" className="icon-button" aria-label="Close recording" onClick={() => (document.getElementById("archive-dialog") as HTMLDialogElement | null)?.close()}>×</button>
            </div>
            <div className="dialog-content">
              <DialogVideo recording={openRecording} />
              <div className="recording-meta">
                <span className="badge">{openRecording.environment}</span>
                <span className="badge">{openRecording.kind}</span>
                <span className="badge">{openRecording.source}</span>
                {openRecording.fitness !== null ? <span className="badge">filename fitness {formatFitness(openRecording.fitness)}</span> : null}
                {openRecording.generation !== null ? <span className="badge">generation {openRecording.generation}</span> : null}
                {openRecording.episode !== null ? <span className="badge">episode {openRecording.episode}</span> : null}
                <span className="badge">{formatBytes(openRecording.bytes)}</span>
              </div>
              <p className="muted">
                Filename fitness is a mean training value, not this clip&apos;s return. Escape closes; focus returns to the card.
              </p>
            </div>
          </>
        ) : null}
      </dialog>
    </section>
  );
}
