"use client";

import { useEffect, useRef, useState } from "react";
import { mediaUrl, type Recording } from "@/lib/data";

export default function Video({ recording, autoPlay = false, controls = false, className = "" }: { recording: Recording; autoPlay?: boolean; controls?: boolean; className?: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [visible, setVisible] = useState(false);
  const [failed, setFailed] = useState(false);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setVisible(true);
        if (autoPlay && !motion.matches) video.play().catch(() => {});
      } else video.pause();
    }, { threshold: 0.15 });
    observer.observe(video);
    const stop = () => { if (motion.matches) video.pause(); };
    motion.addEventListener("change", stop);
    return () => { observer.disconnect(); motion.removeEventListener("change", stop); };
  }, [autoPlay]);
  return (
    <div className={`video-wrap ${className}`}>
      <video ref={ref} src={visible ? mediaUrl(recording) : undefined} muted loop playsInline controls={controls} preload="metadata" onLoadedData={() => { if (autoPlay && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) ref.current?.play().catch(() => {}); }} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onError={() => setFailed(true)} aria-label={`${recording.environment} recorded policy demonstration`} />
      {failed && <p className="video-error">Recording unavailable. Please try again later.</p>}
      {autoPlay && !failed && <button className="video-toggle" aria-label={playing ? "Pause preview" : "Play preview"} onClick={() => { const video = ref.current; if (video?.paused) video.play().catch(() => setFailed(true)); else video?.pause(); }}>{playing ? "Ⅱ" : "▷"}<span>{playing ? "PAUSE" : "PLAY"}</span></button>}
    </div>
  );
}
