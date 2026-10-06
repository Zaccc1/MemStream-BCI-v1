#!/usr/bin/env python3
"""Compare saved pipeline events against the full-recording KS4 cache.

This script mirrors the clean pipeline's KS4 comparison semantics:

- reference spikes come from the KS4 full-recording cache;
- candidate events come from saved software/hardware event shards;
- a match requires time proximity plus spatial proximity;
- exported event shards are corrected by ``--event-time-offset-samples``.

For the current tail3 full-precision shards the required offset is -128
samples, matching clean's FIR_GROUP_DELAY correction in
``spike_clustering.get_results()``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


DEFAULT_KS4_CACHE = Path("/home/wyzhao/git/ks4_cache_full/ks4_full_4267s_384ch.npz")
DEFAULT_CHANNEL_POSITIONS = Path("/home/wyzhao/git/ks4_cache_full/channel_positions.npy")
DEFAULT_EVENTS_ROOT = Path(
    "/home/wyzhao/git/hardware_export/recordings/golden_full_precision/"
    "full_precision_tail3_sharded_0.000s_4266545ms"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/wyzhao/git/hardware_export/analysis/Code_for_analysis/results_ks4_spike_retention"
)


@dataclass
class EventSet:
    name: str
    root: Path
    times: np.ndarray
    channels: np.ndarray
    labels: np.ndarray


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _chunk_start_from_name(path: Path) -> int:
    match = re.search(r"chunks_(\d+)_", path.name)
    return int(match.group(1)) if match else 0


def _event_files(root: Path) -> list[Path]:
    if (root / "tail3_shards").is_dir():
        root = root / "tail3_shards"
    if (root / "actual_events").is_dir():
        root = root / "actual_events"
    files = sorted(root.glob("shard_*_chunks_*.npz"), key=_chunk_start_from_name)
    files += sorted(root.glob("actual_shard_*_chunks_*.npz"), key=_chunk_start_from_name)
    if not files:
        raise FileNotFoundError(f"no event shards found under {root}")
    return files


def _load_events(name: str, root: Path, time_offset_samples: int) -> EventSet:
    time_parts: list[np.ndarray] = []
    channel_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for path in _event_files(root):
        with np.load(path, allow_pickle=False) as z:
            times = np.asarray(z["times"], dtype=np.int64) + int(time_offset_samples)
            channels = np.asarray(z["channels"], dtype=np.int32)
            labels = (
                np.asarray(z["labels"], dtype=np.int32)
                if "labels" in z
                else np.full(times.shape, -1, dtype=np.int32)
            )
        if not (times.shape == channels.shape == labels.shape):
            raise ValueError(f"shape mismatch in {path}")
        time_parts.append(times)
        channel_parts.append(channels)
        label_parts.append(labels)
    times = np.concatenate(time_parts)
    channels = np.concatenate(channel_parts)
    labels = np.concatenate(label_parts)
    order = np.argsort(times, kind="stable")
    return EventSet(name, root, times[order], channels[order], labels[order])


def _load_ks4(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as z:
        times = np.asarray(z["all_times_raw"], dtype=np.int64)
        channels = np.asarray(z["all_channels"], dtype=np.int32)
        labels = np.asarray(z["all_labels"], dtype=np.int32)
        is_ref = np.asarray(z["is_ref"], dtype=bool) if "is_ref" in z else None
    if is_ref is not None:
        good = is_ref[labels]
    else:
        good = np.ones(labels.shape, dtype=bool)
    order = np.argsort(times, kind="stable")
    return {
        "times": times[order],
        "channels": channels[order],
        "labels": labels[order],
        "good": good[order],
    }


def _spatial_hits_ref_to_candidate(
    ref_times: np.ndarray,
    ref_channels: np.ndarray,
    cand_times: np.ndarray,
    cand_channels: np.ndarray,
    positions: np.ndarray,
    sample_rate_hz: float,
    max_dt_ms: float,
    spatial_thresholds_um: list[float],
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    max_dt_s = float(max_dt_ms) / 1000.0
    tree = cKDTree((cand_times.astype(np.float64) / sample_rate_hz).reshape(-1, 1))
    neighbors = tree.query_ball_point(
        (ref_times.astype(np.float64) / sample_rate_hz).reshape(-1, 1),
        r=max_dt_s,
    )
    best_dist = np.full(ref_times.shape, np.inf, dtype=np.float32)
    for i, inds in enumerate(neighbors):
        if not inds:
            continue
        cidx = np.asarray(inds, dtype=np.int64)
        cpos = positions[cand_channels[cidx]]
        rpos = positions[int(ref_channels[i])]
        dists = np.sqrt(np.sum((cpos - rpos) ** 2, axis=1))
        best_dist[i] = float(np.min(dists))
    return {th: best_dist < float(th) for th in spatial_thresholds_um}, best_dist


def _evaluate_window(
    *,
    ks4: dict[str, np.ndarray],
    events: EventSet,
    positions: np.ndarray,
    start_sample: int,
    stop_sample: int,
    sample_rate_hz: float,
    max_dt_ms: float,
    spatial_thresholds_um: list[float],
    accepted_only: bool,
) -> dict[str, Any]:
    kmask = (ks4["times"] >= start_sample) & (ks4["times"] < stop_sample)
    emask = (events.times >= start_sample) & (events.times < stop_sample)
    if accepted_only:
        emask &= events.labels >= 0

    ref_times = ks4["times"][kmask]
    ref_channels = ks4["channels"][kmask]
    ref_good = ks4["good"][kmask]
    cand_times = events.times[emask]
    cand_channels = events.channels[emask]

    out: dict[str, Any] = {
        "start_sample": int(start_sample),
        "stop_sample": int(stop_sample),
        "start_s": float(start_sample / sample_rate_hz),
        "stop_s": float(stop_sample / sample_rate_hz),
        "ks4_all": int(ref_times.size),
        "ks4_good": int(np.count_nonzero(ref_good)),
        "candidate_events": int(cand_times.size),
        "accepted_only": bool(accepted_only),
        "thresholds": {},
    }
    if ref_times.size == 0 or cand_times.size == 0:
        for th in spatial_thresholds_um:
            out["thresholds"][str(th)] = {
                "recall_all": 0.0,
                "recall_good": 0.0,
                "matched_all": 0,
                "matched_good": 0,
            }
        return out

    hits_by_thresh, _ = _spatial_hits_ref_to_candidate(
        ref_times,
        ref_channels,
        cand_times,
        cand_channels,
        positions,
        sample_rate_hz,
        max_dt_ms,
        spatial_thresholds_um,
    )
    n_good = int(np.count_nonzero(ref_good))
    for th, hits in hits_by_thresh.items():
        matched_all = int(np.count_nonzero(hits))
        matched_good = int(np.count_nonzero(hits & ref_good))
        out["thresholds"][str(th)] = {
            "recall_all": matched_all / ref_times.size if ref_times.size else 0.0,
            "recall_good": matched_good / n_good if n_good else 0.0,
            "matched_all": matched_all,
            "matched_good": matched_good,
        }
    return out


def _combine_windows(rows: list[dict[str, Any]], spatial_thresholds_um: list[float]) -> dict[str, Any]:
    total_all = sum(int(r["ks4_all"]) for r in rows)
    total_good = sum(int(r["ks4_good"]) for r in rows)
    total_cand = sum(int(r["candidate_events"]) for r in rows)
    combined = {
        "ks4_all": int(total_all),
        "ks4_good": int(total_good),
        "candidate_events": int(total_cand),
        "thresholds": {},
    }
    for th in spatial_thresholds_um:
        key = str(th)
        matched_all = sum(int(r["thresholds"][key]["matched_all"]) for r in rows)
        matched_good = sum(int(r["thresholds"][key]["matched_good"]) for r in rows)
        combined["thresholds"][key] = {
            "recall_all": matched_all / total_all if total_all else 0.0,
            "recall_good": matched_good / total_good if total_good else 0.0,
            "matched_all": int(matched_all),
            "matched_good": int(matched_good),
        }
    return combined


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# KS4 Spike Retention")
    lines.append("")
    lines.append(f"KS4 cache: `{summary['ks4_cache']}`")
    lines.append(f"events root: `{summary['events_root']}`")
    lines.append(f"event_time_offset_samples: `{summary['event_time_offset_samples']}`")
    lines.append(f"sample_rate_hz: `{summary['sample_rate_hz']}`")
    lines.append(f"max_dt_ms: `{summary['max_dt_ms']}`")
    lines.append("")
    lines.append("| subset | spatial_um | KS4 all recall | KS4 good recall | candidate events |")
    lines.append("|---|---:|---:|---:|---:|")
    for subset_name in ["detected", "accepted"]:
        combined = summary[subset_name]["combined"]
        for th, vals in combined["thresholds"].items():
            lines.append(
                f"| {subset_name} | {th} | "
                f"{vals['matched_all']}/{combined['ks4_all']} = {vals['recall_all']:.4f} | "
                f"{vals['matched_good']}/{combined['ks4_good']} = {vals['recall_good']:.4f} | "
                f"{combined['candidate_events']} |"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks4-cache", type=Path, default=DEFAULT_KS4_CACHE)
    parser.add_argument("--channel-positions", type=Path, default=DEFAULT_CHANNEL_POSITIONS)
    parser.add_argument("--events-root", type=Path, default=DEFAULT_EVENTS_ROOT)
    parser.add_argument("--name", default="pipeline_events")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate-hz", type=float, default=30000.0)
    parser.add_argument("--max-dt-ms", type=float, default=0.5)
    parser.add_argument("--spatial-thresholds-um", default="50,100,150")
    parser.add_argument(
        "--event-time-offset-samples",
        type=int,
        default=-128,
        help="Apply to saved event times before matching. Current tail3 shards need -128.",
    )
    parser.add_argument("--start-s", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument(
        "--window-s",
        type=float,
        default=30.0,
        help="Process in windows to keep memory and KD-tree sizes bounded.",
    )
    args = parser.parse_args()

    thresholds = [float(x) for x in args.spatial_thresholds_um.split(",") if x.strip()]
    positions = np.load(args.channel_positions, allow_pickle=False)
    ks4 = _load_ks4(args.ks4_cache)
    events = _load_events(args.name, args.events_root, args.event_time_offset_samples)

    data_start = max(int(ks4["times"].min()), int(events.times.min()))
    data_stop = min(int(ks4["times"].max()) + 1, int(events.times.max()) + 1)
    if args.start_s is not None:
        data_start = max(data_start, int(round(args.start_s * args.sample_rate_hz)))
    if args.duration_s is not None:
        data_stop = min(data_stop, data_start + int(round(args.duration_s * args.sample_rate_hz)))
    if data_stop <= data_start:
        raise ValueError(f"empty overlap: {data_start}..{data_stop}")

    window_samples = max(1, int(round(args.window_s * args.sample_rate_hz)))
    subset_rows: dict[str, list[dict[str, Any]]] = {"detected": [], "accepted": []}
    start = data_start
    while start < data_stop:
        stop = min(data_stop, start + window_samples)
        for subset_name, accepted_only in [("detected", False), ("accepted", True)]:
            subset_rows[subset_name].append(
                _evaluate_window(
                    ks4=ks4,
                    events=events,
                    positions=positions,
                    start_sample=start,
                    stop_sample=stop,
                    sample_rate_hz=args.sample_rate_hz,
                    max_dt_ms=args.max_dt_ms,
                    spatial_thresholds_um=thresholds,
                    accepted_only=accepted_only,
                )
            )
        start = stop

    summary: dict[str, Any] = {
        "ks4_cache": args.ks4_cache,
        "channel_positions": args.channel_positions,
        "events_root": args.events_root,
        "event_name": args.name,
        "event_time_offset_samples": int(args.event_time_offset_samples),
        "sample_rate_hz": float(args.sample_rate_hz),
        "max_dt_ms": float(args.max_dt_ms),
        "spatial_thresholds_um": thresholds,
        "range_samples": [int(data_start), int(data_stop)],
        "detected": {
            "combined": _combine_windows(subset_rows["detected"], thresholds),
            "windows": subset_rows["detected"],
        },
        "accepted": {
            "combined": _combine_windows(subset_rows["accepted"], thresholds),
            "windows": subset_rows["accepted"],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "ks4_spike_retention_summary.json", summary)
    _write_report(args.output_dir / "ks4_spike_retention_report.md", summary)

    rows_path = args.output_dir / "ks4_spike_retention_windows.csv"
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subset",
                "start_s",
                "stop_s",
                "spatial_um",
                "ks4_all",
                "ks4_good",
                "candidate_events",
                "matched_all",
                "matched_good",
                "recall_all",
                "recall_good",
            ],
        )
        writer.writeheader()
        for subset_name in ["detected", "accepted"]:
            for row in subset_rows[subset_name]:
                for th, vals in row["thresholds"].items():
                    writer.writerow(
                        {
                            "subset": subset_name,
                            "start_s": row["start_s"],
                            "stop_s": row["stop_s"],
                            "spatial_um": th,
                            "ks4_all": row["ks4_all"],
                            "ks4_good": row["ks4_good"],
                            "candidate_events": row["candidate_events"],
                            "matched_all": vals["matched_all"],
                            "matched_good": vals["matched_good"],
                            "recall_all": vals["recall_all"],
                            "recall_good": vals["recall_good"],
                        }
                    )

    print(f"summary -> {args.output_dir / 'ks4_spike_retention_summary.json'}")
    print(f"report  -> {args.output_dir / 'ks4_spike_retention_report.md'}")
    print(f"windows -> {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
