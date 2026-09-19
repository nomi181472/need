"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { environmentsWithDisplay, mediaUrl, recordings, featuredRecording } from "@/lib/data";
import type { Recording } from "@/lib/data";

function RecordingPicker({
  id,
  label,
  value,
  environment,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  environment: string;
  onChange: (id: string) => void;
}) {
  const options = useMemo(
    () => recordings.filter((recording) => recording.environment === environment),
    [environment],
  );
  return (
    <>
      <label className="muted" htmlFor={id}>{label}</label>
      <select
        id={id}
        className="select-input"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.length === 0 ? <option value="">No recordings</option> : null}
        {options.map((recording) => (
          <option key={recording.id} value={recording.id}>
            {recording.filename}
          </option>
        ))}
      </select>
    </>
  );
}

function LinkedVideo({ recording }: { recording: Recording | null }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onEnded = () => setPlaying(false);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("ended", onEnded);
    return () => {
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("ended", onEnded);
    };
  }, [recording]);

  const toggle = async () => {
    const video = videoRef.current;
    if (!video) return;
    try {
      if (video.paused) {
        await video.play();
      } else {
        video.pause();
      }
    } catch {
      setFailed(true);
    }
  };

  const restart = async () => {
    const video = videoRef.current;
    if (!video) return;
    try {
      video.currentTime = 0;
      await video.play();
    } catch {
      setFailed(true);
    }
  };

  if (!recording) {
    return (
      <div className="comparison-panel">
        <div className="empty-state">
          <p className="stat-label">Nothing selected</p>
          <p className="muted">Pick a recording to load a player.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="comparison-panel">
      {failed ? (
        <div className="empty-state">
          <p className="stat-label">Playback failed</p>
          <p className="muted">The browser refused to start this video. Try the native controls.</p>
          <a className="text-link" href={mediaUrl(recording)} target="_blank" rel="noreferrer">Download recording</a>
        </div>
      ) : null}
      <video
        ref={videoRef}
        src={mediaUrl(recording)}
        controls
        muted
        preload="none"
        playsInline
        style={{ width: "100%", height: "auto", display: "block", background: "#000" }}
      />
      <div className="recording-body">
        <p className="recording-meta">{recording.filename}</p>
        <div className="recording-meta">
          <button type="button" className="button button-ghost" onClick={toggle}>
            {playing ? "Pause" : "Play"}
          </button>
          <button type="button" className="button button-ghost" onClick={restart}>Restart</button>
        </div>
      </div>
    </div>
  );
}

export default function Comparison() {
  const environments = environmentsWithDisplay();
  const [envA, setEnvA] = useState(environments[0]?.id ?? "");
  const [envB, setEnvB] = useState(environments[1]?.id ?? environments[0]?.id ?? "");
  const [pickA, setPickA] = useState(() => featuredRecording(envA)?.id ?? "");
  const [pickB, setPickB] = useState(() => featuredRecording(envB)?.id ?? "");

  const poolA = useMemo(() => recordings.filter((recording) => recording.environment === envA), [envA]);
  const poolB = useMemo(() => recordings.filter((recording) => recording.environment === envB), [envB]);

  const recordingA = poolA.find((recording) => recording.id === pickA) ?? null;
  const recordingB = poolB.find((recording) => recording.id === pickB) ?? null;

  return (
    <section aria-label="Recording comparison">
      <div className="comparison-controls">
        <label className="muted" htmlFor="env-a">Environment left</label>
        <select
          id="env-a"
          className="select-input"
          value={envA}
          onChange={(event) => {
            const environment = event.target.value;
            setEnvA(environment);
            setPickA(featuredRecording(environment)?.id ?? "");
          }}
        >
          {environments.map((environment) => (
            <option key={environment.id} value={environment.id}>{environment.name}</option>
          ))}
        </select>
        <RecordingPicker id="pick-a" label="Recording left" value={pickA} environment={envA} onChange={setPickA} />
        <label className="muted" htmlFor="env-b">Environment right</label>
        <select
          id="env-b"
          className="select-input"
          value={envB}
          onChange={(event) => {
            const environment = event.target.value;
            setEnvB(environment);
            setPickB(featuredRecording(environment)?.id ?? "");
          }}
        >
          {environments.map((environment) => (
            <option key={environment.id} value={environment.id}>{environment.name}</option>
          ))}
        </select>
        <RecordingPicker id="pick-b" label="Recording right" value={pickB} environment={envB} onChange={setPickB} />
      </div>

      <div className="comparison-grid">
        <LinkedVideo key={`left-${envA}-${pickA}`} recording={recordingA} />
        <LinkedVideo key={`right-${envB}-${pickB}`} recording={recordingB} />
      </div>

      <p className="muted">
        Players run independently with native controls. Different episode lengths are normal here — the archive does not lock frames across recordings.
      </p>
    </section>
  );
}
