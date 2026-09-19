#!/usr/bin/env python3
"""Extract learning-curve scalars from v1/* tfevents logs into portfolio/content/curves.json.

One-time preprocessing. Reads the TensorFlow event files committed under v1/ (v1/logs,
v1/logs_archive_2, v1/lunar_lander_logs, v1/preserved_logs_2), groups them into runs, and
emits a small JSON the Next.js portfolio reads at build time.

Usage:
    python3 scripts/extract-curves.py [--inspect]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing import event_file_loader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
CONTENT_DIR = SCRIPT_DIR.parent / "content"

ENV_LOGS: dict[str, list[tuple[str, str | None]]] = {
    "Ant-v5": [("v1/logs/Ant-v5", None)],
    "HumanoidStandup-v5": [("v1/logs/HumanoidStandup-v5", None)],
    "HalfCheetah-v5": [("v1/preserved_logs_2/halfcheetah", None)],
    "Walker2d-v5": [("v1/preserved_logs_2/walker2d", None)],
    "Swimmer-v5": [
        ("v1/preserved_logs_2/swimmer_logs_final", None),
        ("v1/preserved_logs_2/swimmer", "Swimmer"),
    ],
    "LunarLander-v3": [
        ("v1/lunar_lander_logs", None),
        ("v1/preserved_logs_2/lunarlander", None),
        ("v1/preserved_logs_2/logs_archive_2", None),
        ("v1/logs_archive_2/lunar_lander_deteministic", None),
        ("v1/preserved_logs_2/swimmer", "LunarLander"),
    ],
    "CartPole-v1": [("v1/preserved_logs_2/cartpole-grid-search", None)],
}

EVENT_RE = re.compile(r"events\.out\.tfevents\.(\d+)\.([^./]+)\.(\d+)(?:\.(\d+))?$")

# normalized key -> list of source tags tried in order
SERIES_TAGS: dict[str, list[str]] = {
    "best_reward": ["main_metrics/best_reward", "main_metrics/pop_best", "population/max_reward"],
    "mean_reward": ["main_metrics/mean_reward", "population/mean_reward"],
    "median_reward": ["main_metrics/median_reward", "population/median_reward"],
    "coverage": ["qd_metrics/coverage_percentage", "qd_metrics/coverage", "main_metrics/coverage"],
    "qd_score": ["main_metrics/qd_score"],
}
MIN_STEPS = 5


def _seq(fname: str) -> int | None:
    m = EVENT_RE.search(os.path.basename(fname))
    if not m:
        return None
    return int(m.group(4) or 0)


def _parse(fname: str) -> tuple[str, str] | None:
    m = EVENT_RE.search(os.path.basename(fname))
    if not m:
        return None
    return m.group(2), m.group(3)  # host, pid


def discover_files(env_roots: list[tuple[str, str | None]]) -> list[str]:
    files = []
    for root, name_filter in env_roots:
        base = os.path.join(REPO_ROOT, root) if not os.path.isabs(root) else root
        for f in glob.glob(os.path.join(base, "**", "events.out.tfevents.*"), recursive=True):
            parent = os.path.basename(os.path.dirname(f))
            if name_filter and name_filter not in parent:
                continue
            files.append(f)
    return files


def group_runs(files: list[str]) -> list[list[str]]:
    """Group event files into runs.

    tfevents writers shard into files (a top-level file plus a descendant `logs/<name>/`
    subtree). Different experiment harnesses reuse the same OS pid across many runs, so we
    first bucket by the parent directory name, then split that bucket by (host, pid). This
    keeps Ant/HumanoidStandup's multiple seed writers distinct while separating the Walker2d
    and CartPole operator-grid runs that share one pid.
    """
    by_parent: dict[str, list[str]] = defaultdict(list)
    for f in files:
        parent = os.path.basename(os.path.dirname(f))
        by_parent[parent].append(f)

    runs: list[list[str]] = []
    for parent, fs in by_parent.items():
        by_pid: dict[str, list[str]] = defaultdict(list)
        for f in fs:
            key = _parse(f)
            by_pid[f"{key[0]}.{key[1]}" if key else "?"].append(f)
        for pid, pid_files in by_pid.items():
            key = None
            pid_files.sort(key=lambda f: int(EVENT_RE.search(os.path.basename(f)).group(1)))
            # seq-based shards within a writer may reuse ts; keep ts-ordered, split on reset
            run: list[str] = [pid_files[0]]
            for prev, cur in zip(pid_files, pid_files[1:]):
                prev_ts = int(EVENT_RE.search(os.path.basename(prev)).group(1))
                cur_ts = int(EVENT_RE.search(os.path.basename(cur)).group(1))
                prev_seq = _seq(prev) or 0
                cur_seq = _seq(cur) or 0
                if cur_seq <= prev_seq and cur_ts - prev_ts > 600:
                    runs.append(run)
                    run = []
                run.append(cur)
            runs.append(run)
    return runs


def run_label(files: list[str], env_id: str) -> str:
    names = []
    seen = set()
    for f in files:
        parent = os.path.basename(os.path.dirname(f))
        if parent in ("logs", env_id) or parent in seen:
            continue
        seen.add(parent)
        names.append(parent)
    if names:
        return names[0]
    m = _parse(files[0])
    return f"run {m[1]}" if m else "run"


def display_name(label: str) -> str:
    """Condense raw grid-run directory names into short human labels.

    Examples:
      grid_polynomial_uniform_run_0_layers_2_rate_power_replace_1_1_1 -> "polynomial uniform · 2 layers"
      grid_adaptive_blend                                                -> "adaptive blend"
      grid_gaussian_sbx_run0                                             -> "gaussian sbx"
    """
    if not label.startswith("grid_"):
        return label
    rest = label[len("grid_"):]
    layers = None
    if "_run_0_" in rest:
        base, tail = rest.split("_run_0_", 1)
        m = re.search(r"layers_(\d+)", tail)
        if m:
            layers = int(m.group(1))
    elif rest.endswith("_run0"):
        base = rest[: -len("_run0")].rstrip("_")
    else:
        base = rest
    cleaned = " ".join(base.split("_"))
    return f"{cleaned} · {layers} layers" if layers else cleaned


def read_run(files: list[str]) -> dict[str, list[tuple[int, float]]]:
    by_tag: dict[str, dict[int, float]] = defaultdict(dict)
    for f in files:
        try:
            loader = event_file_loader.LegacyEventFileLoader(f)
            for ev in loader.Load():
                if ev.WhichOneof("what") != "summary":
                    continue
                for value in ev.summary.value:
                    if value.tag not in by_tag:
                        by_tag[value.tag] = {}
                    v = value.simple_value if value.HasField("simple_value") else None
                    if v is None and value.HasField("tensor"):
                        t = value.tensor
                        import struct as _s

                        try:
                            float_list = t.DoubleVal or list(_s.iter_unpack("f", t.TensorContent or b""))
                            v = float_list[0] if float_list else None
                        except Exception:
                            v = None
                    if v is None:
                        continue
                    by_tag[value.tag][ev.step] = v
        except Exception as ex:  # noqa: BLE001
            print(f"  warn: unreadable {os.path.basename(f)}: {ex}")
    series: dict[str, list[tuple[int, float]]] = {}
    for key, tags in SERIES_TAGS.items():
        for tag in tags:
            if tag not in by_tag:
                continue
            rows = sorted((s, v) for s, v in by_tag[tag].items())
            if len(rows) < MIN_STEPS:
                continue
            # some harnesses log a degenerate all-zero summary tag; skip to a real series
            if max(v for _, v in rows) == 0.0 and min(v for _, v in rows) == 0.0:
                continue
            series[key] = rows
            break
    return series


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="print per-env/run stats instead of writing JSON")
    args = ap.parse_args(argv)

    out: dict[str, object] = {"environments": {}}
    total_runs = 0
    for env_id, roots in ENV_LOGS.items():
        files = discover_files(roots)
        runs = group_runs(files)
        valid = []
        for files_of_run in runs:
            label = run_label(files_of_run, env_id)
            series = read_run(files_of_run)
            if not series:
                continue
            valid.append({"name": label, "files": len(files_of_run), "series": series})
            total_runs += 1

        # coverage normalization to percent when stored as a fraction
        for r in valid:
            for key in ("coverage",):
                rows = r["series"].get(key)
                if rows and max(v for _, v in rows) <= 1.0001:
                    r["series"][key] = [(s, round(v * 100.0, 2)) for s, v in rows]

        # sort: named grid runs keep dir order; then alphabetically
        named = sorted([r for r in valid if not r["name"].startswith("run ")], key=lambda r: r["name"])
        unnamed = sorted([r for r in valid if r["name"].startswith("run ")], key=lambda r: (r["name"], r["files"]))
        ordered = named + unnamed

        if args.inspect:
            print(f"\n## {env_id}: {len(ordered)} runs kept of {len(runs)} discovered")
            for r in ordered:
                detail = ", ".join(f"{k}={len(rows)}pts" for k, rows in r["series"].items())
                first = r["series"].get("best_reward")
                span = f"[gen {first[0][0]}..{first[-1][0]} best {first[-1][1]:.1f}]" if first else ""
                print(f"   {r['name'][:48]:48s} files={r['files']:3d} {detail:55s} {span}")
            continue

        compact = []
        for r in ordered:
            entry = {"name": display_name(r["name"]), "series": {}}
            for key, rows in r["series"].items():
                entry["series"][key] = [[int(s), round(v, 2)] for s, v in rows]
            compact.append(entry)
        out["environments"][env_id] = {"runs": compact}  # type: ignore[assignment]

    if args.inspect:
        return

    out["generated"] = _dt.date.today().isoformat()
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    target = CONTENT_DIR / "curves.json"
    target.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    sizes = {
        env: sum(len(r["series"]["best_reward"]) for r in val["runs"] if "best_reward" in r["series"])
        for env, val in out["environments"].items()
    }
    print(f"wrote {target} ({target.stat().st_size/1024:.0f} KB)")
    print(f"runs: {total_runs}")
    print("best_reward points per env:", sizes)


if __name__ == "__main__":
    main()