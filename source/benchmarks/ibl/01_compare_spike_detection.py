#!/usr/bin/env python3
"""Compare spike-detection event timing/channel outputs across saved sources.

Default sources:
- full precision software golden
- 8-bit software golden
- 8-bit hardware saved events

The script matches events by sample time and channel, then reports pairwise
precision/recall/F1 plus per-source event counts. By default it restricts all
sources to their common available time range, which avoids penalizing a partial
hardware run for chunks that were not streamed yet.
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


DEFAULT_FULL_ROOT = Path(
    "/home/wyzhao/git/hardware_export/recordings/golden_full_precision/"
    "full_precision_tail3_sharded_0.000s_4266545ms"
)
DEFAULT_SW8_ROOT = Path(
    "/home/wyzhao/git/6.20_TEST1/data/"
    "clean_recording_tail3_sharded_0.000s_4266545ms"
)
DEFAULT_HW8_ROOT = Path(
    "/home/wyzhao/git/hardware_export/analysis/vmm/"
    "full_recording_hw12_whitegate4_topkfix_cal5s_streamrest_20260626/"
    "actual_events"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/wyzhao/git/hardware_export/analysis/Code_for_analysis/results_spike_detection"
)


@dataclass
class EventSet:
    name: str
    root: Path
    times: np.ndarray
    channels: np.ndarray
    labels: np.ndarray
    chunk_start: int | None
    chunk_stop: int | None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _chunk_start_from_name(path: Path) -> int:
    m = re.search(r"chunks_(\d+)_", path.name)
    if m:
        return int(m.group(1))
    return 0


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


def _load_source(
    name: str,
    root: Path,
    *,
    chunk_len: int,
    chunk_start: int | None,
    chunk_stop: int | None,
) -> EventSet:
    time_parts: list[np.ndarray] = []
    channel_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    min_chunk: int | None = None
    max_chunk: int | None = None
    for path in _event_files(root):
        with np.load(path, allow_pickle=False) as z:
            local_start = int(z["chunk_start"]) if "chunk_start" in z else _chunk_start_from_name(path)
            local_count = int(z["chunk_count"]) if "chunk_count" in z else 1000
            local_stop = local_start + local_count
            if chunk_start is not None and local_stop <= chunk_start:
                continue
            if chunk_stop is not None and local_start >= chunk_stop:
                continue
            times = np.asarray(z["times"], dtype=np.int64)
            channels = np.asarray(z["channels"], dtype=np.int16)
            labels = (
                np.asarray(z["labels"], dtype=np.int32)
                if "labels" in z
                else np.full(times.shape, -1, dtype=np.int32)
            )
        if not (times.shape == channels.shape == labels.shape):
            raise ValueError(f"shape mismatch in {path}: {times.shape}, {channels.shape}, {labels.shape}")
        keep = np.ones(times.shape, dtype=bool)
        if chunk_start is not None:
            keep &= times >= int(chunk_start) * int(chunk_len)
        if chunk_stop is not None:
            keep &= times < int(chunk_stop) * int(chunk_len)
        if not np.any(keep):
            continue
        time_parts.append(times[keep])
        channel_parts.append(channels[keep])
        label_parts.append(labels[keep])
        min_chunk = local_start if min_chunk is None else min(min_chunk, local_start)
        max_chunk = local_stop if max_chunk is None else max(max_chunk, local_stop)
    if not time_parts:
        raise ValueError(f"no events loaded for {name} from {root}")
    times = np.concatenate(time_parts)
    channels = np.concatenate(channel_parts)
    labels = np.concatenate(label_parts)
    order = np.lexsort((channels, times))
    return EventSet(
        name=name,
        root=root,
        times=times[order],
        channels=channels[order],
        labels=labels[order],
        chunk_start=min_chunk,
        chunk_stop=max_chunk,
    )


def _filter_time_range(src: EventSet, start_sample: int, stop_sample: int) -> EventSet:
    keep = (src.times >= start_sample) & (src.times < stop_sample)
    return EventSet(
        name=src.name,
        root=src.root,
        times=src.times[keep],
        channels=src.channels[keep],
        labels=src.labels[keep],
        chunk_start=start_sample // 300,
        chunk_stop=(stop_sample + 299) // 300,
    )


def _match_exact_channel(
    ref: EventSet,
    cand: EventSet,
    *,
    time_tol: int,
) -> tuple[np.ndarray, np.ndarray]:
    ref_matches: list[np.ndarray] = []
    cand_matches: list[np.ndarray] = []
    for ch in np.intersect1d(np.unique(ref.channels), np.unique(cand.channels)):
        ri = np.flatnonzero(ref.channels == ch)
        ci = np.flatnonzero(cand.channels == ch)
        if ri.size == 0 or ci.size == 0:
            continue
        ro = ri[np.argsort(ref.times[ri], kind="stable")]
        co = ci[np.argsort(cand.times[ci], kind="stable")]
        i = 0
        j = 0
        local_r: list[int] = []
        local_c: list[int] = []
        while i < ro.size and j < co.size:
            rt = int(ref.times[ro[i]])
            ct = int(cand.times[co[j]])
            if ct < rt - time_tol:
                j += 1
            elif ct > rt + time_tol:
                i += 1
            else:
                local_r.append(int(ro[i]))
                local_c.append(int(co[j]))
                i += 1
                j += 1
        if local_r:
            ref_matches.append(np.asarray(local_r, dtype=np.int64))
            cand_matches.append(np.asarray(local_c, dtype=np.int64))
    if not ref_matches:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(ref_matches), np.concatenate(cand_matches)


def _match_with_channel_tol(
    ref: EventSet,
    cand: EventSet,
    *,
    time_tol: int,
    channel_tol: int,
) -> tuple[np.ndarray, np.ndarray]:
    if channel_tol <= 0:
        return _match_exact_channel(ref, cand, time_tol=time_tol)
    cand_order = np.argsort(cand.times, kind="stable")
    cand_times = cand.times[cand_order]
    used = np.zeros(cand.times.shape, dtype=bool)
    ref_out: list[int] = []
    cand_out: list[int] = []
    ref_order = np.argsort(ref.times, kind="stable")
    for ri in ref_order.tolist():
        rt = int(ref.times[ri])
        rc = int(ref.channels[ri])
        best_ci = -1
        best_key = (time_tol + 1, channel_tol + 1)
        lo = int(np.searchsorted(cand_times, rt - time_tol, side="left"))
        hi = int(np.searchsorted(cand_times, rt + time_tol, side="right"))
        for jj in range(lo, hi):
            ci = int(cand_order[jj])
            if used[ci]:
                continue
            dc = abs(int(cand.channels[ci]) - rc)
            if dc > channel_tol:
                continue
            dt = abs(int(cand.times[ci]) - rt)
            if (dt, dc) < best_key:
                best_ci = ci
                best_key = (dt, dc)
        if best_ci >= 0:
            used[best_ci] = True
            ref_out.append(ri)
            cand_out.append(best_ci)
    return np.asarray(ref_out, dtype=np.int64), np.asarray(cand_out, dtype=np.int64)


def _event_count_by_chunk(src: EventSet, chunk_len: int) -> dict[str, float | int | None]:
    if src.times.size == 0:
        return {"chunks_with_events": 0, "mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}
    chunks, counts = np.unique(src.times // int(chunk_len), return_counts=True)
    return {
        "chunks_with_events": int(chunks.size),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "p05": float(np.percentile(counts, 5)),
        "p95": float(np.percentile(counts, 95)),
    }


def _source_summary(src: EventSet, chunk_len: int) -> dict[str, Any]:
    return {
        "name": src.name,
        "root": src.root,
        "event_count": int(src.times.size),
        "accepted_count": int(np.count_nonzero(src.labels >= 0)),
        "min_time_sample": int(src.times.min()) if src.times.size else None,
        "max_time_sample": int(src.times.max()) if src.times.size else None,
        "chunk_start": src.chunk_start,
        "chunk_stop": src.chunk_stop,
        "per_chunk": _event_count_by_chunk(src, chunk_len),
    }


def _pair_metrics(ref: EventSet, cand: EventSet, *, time_tol: int, channel_tol: int) -> dict[str, Any]:
    ri, ci = _match_with_channel_tol(ref, cand, time_tol=time_tol, channel_tol=channel_tol)
    matched = int(ri.size)
    recall = matched / int(ref.times.size) if ref.times.size else None
    precision = matched / int(cand.times.size) if cand.times.size else None
    f1 = None
    if recall is not None and precision is not None and (recall + precision) > 0:
        f1 = 2.0 * recall * precision / (recall + precision)
    dt = np.abs(ref.times[ri] - cand.times[ci]) if matched else np.zeros(0, dtype=np.int64)
    dc = np.abs(ref.channels[ri].astype(np.int32) - cand.channels[ci].astype(np.int32)) if matched else np.zeros(0)
    return {
        "reference": ref.name,
        "candidate": cand.name,
        "reference_events": int(ref.times.size),
        "candidate_events": int(cand.times.size),
        "matched": matched,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "time_abs_diff_mean": float(np.mean(dt)) if matched else None,
        "time_abs_diff_p95": float(np.percentile(dt, 95)) if matched else None,
        "channel_abs_diff_mean": float(np.mean(dc)) if matched else None,
        "channel_abs_diff_p95": float(np.percentile(dc, 95)) if matched else None,
    }


def _write_pair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "reference",
        "candidate",
        "reference_events",
        "candidate_events",
        "matched",
        "recall",
        "precision",
        "f1",
        "time_abs_diff_mean",
        "time_abs_diff_p95",
        "channel_abs_diff_mean",
        "channel_abs_diff_p95",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Spike Detection Comparison",
        "",
        f"time_tol_samples: `{summary['time_tol_samples']}`",
        f"channel_tol: `{summary['channel_tol']}`",
        f"common_range_samples: `{summary['common_range_samples']}`",
        "",
        "## Sources",
        "",
        "| source | events | accepted | min_sample | max_sample |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["sources"]:
        lines.append(
            f"| {item['name']} | {item['event_count']} | {item['accepted_count']} | "
            f"{item['min_time_sample']} | {item['max_time_sample']} |"
        )
    lines += [
        "",
        "## Pairwise Detection",
        "",
        "| reference | candidate | matched | recall | precision | f1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["pairs"]:
        lines.append(
            f"| {row['reference']} | {row['candidate']} | {row['matched']} | "
            f"{(row['recall'] or 0.0):.4f} | {(row['precision'] or 0.0):.4f} | "
            f"{(row['f1'] or 0.0):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    p.add_argument("--sw8-root", type=Path, default=DEFAULT_SW8_ROOT)
    p.add_argument("--hw8-root", type=Path, default=DEFAULT_HW8_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--chunk-len", type=int, default=300)
    p.add_argument("--chunk-start", type=int, default=None)
    p.add_argument("--chunk-stop", type=int, default=None)
    p.add_argument("--time-tol-samples", type=int, default=5)
    p.add_argument("--channel-tol", type=int, default=0)
    p.add_argument(
        "--benchmark-source",
        choices=["full_precision", "sw8_software", "hw8_hardware"],
        default=None,
        help="If set, compute only benchmark -> other-source pairs instead of all ordered pairs.",
    )
    p.add_argument("--no-common-range", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sources = [
        _load_source("full_precision", args.full_root, chunk_len=args.chunk_len, chunk_start=args.chunk_start, chunk_stop=args.chunk_stop),
        _load_source("sw8_software", args.sw8_root, chunk_len=args.chunk_len, chunk_start=args.chunk_start, chunk_stop=args.chunk_stop),
        _load_source("hw8_hardware", args.hw8_root, chunk_len=args.chunk_len, chunk_start=args.chunk_start, chunk_stop=args.chunk_stop),
    ]
    common_range: list[int | None] = [None, None]
    if not args.no_common_range:
        start = max(int(s.times.min()) for s in sources if s.times.size)
        stop = min(int(s.times.max()) + 1 for s in sources if s.times.size)
        common_range = [start, stop]
        sources = [_filter_time_range(s, start, stop) for s in sources]

    pairs = []
    if args.benchmark_source is not None:
        by_name = {s.name: s for s in sources}
        ref = by_name[args.benchmark_source]
        for cand in sources:
            if cand.name == ref.name:
                continue
            pairs.append(_pair_metrics(ref, cand, time_tol=args.time_tol_samples, channel_tol=args.channel_tol))
    else:
        for i, ref in enumerate(sources):
            for j, cand in enumerate(sources):
                if i == j:
                    continue
                pairs.append(_pair_metrics(ref, cand, time_tol=args.time_tol_samples, channel_tol=args.channel_tol))

    summary = {
        "format": "spike_detection_compare.v1",
        "time_tol_samples": int(args.time_tol_samples),
        "channel_tol": int(args.channel_tol),
        "chunk_len": int(args.chunk_len),
        "requested_chunk_start": args.chunk_start,
        "requested_chunk_stop": args.chunk_stop,
        "benchmark_source": args.benchmark_source,
        "common_range_samples": common_range,
        "sources": [_source_summary(s, args.chunk_len) for s in sources],
        "pairs": pairs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "spike_detection_summary.json", summary)
    _write_pair_csv(args.output_dir / "spike_detection_pairs.csv", pairs)
    _write_report(args.output_dir / "spike_detection_report.md", summary)
    print(f"summary -> {args.output_dir / 'spike_detection_summary.json'}")
    print(f"report  -> {args.output_dir / 'spike_detection_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
