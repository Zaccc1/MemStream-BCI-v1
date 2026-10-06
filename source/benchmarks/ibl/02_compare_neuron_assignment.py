#!/usr/bin/env python3
"""Compare neuron-assignment labels after time/channel event matching.

The default reference is the 8-bit software golden. The script matches each
candidate source to the reference by event time/channel, then reports detection
match metrics plus assignment metrics on matched reference-accepted events.
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
    "/home/wyzhao/git/hardware_export/analysis/Code_for_analysis/results_neuron_assignment"
)


@dataclass
class EventSet:
    name: str
    root: Path
    times: np.ndarray
    channels: np.ndarray
    labels: np.ndarray
    distances: np.ndarray


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
    return int(m.group(1)) if m else 0


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
    distance_parts: list[np.ndarray] = []
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
            distances = (
                np.asarray(z["distances"], dtype=np.float32)
                if "distances" in z
                else np.full(times.shape, np.inf, dtype=np.float32)
            )
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
        distance_parts.append(distances[keep])
    if not time_parts:
        raise ValueError(f"no events loaded for {name} from {root}")
    times = np.concatenate(time_parts)
    channels = np.concatenate(channel_parts)
    labels = np.concatenate(label_parts)
    distances = np.concatenate(distance_parts)
    order = np.lexsort((channels, times))
    return EventSet(name, root, times[order], channels[order], labels[order], distances[order])


def _filter_time_range(src: EventSet, start_sample: int, stop_sample: int) -> EventSet:
    keep = (src.times >= start_sample) & (src.times < stop_sample)
    return EventSet(
        src.name,
        src.root,
        src.times[keep],
        src.channels[keep],
        src.labels[keep],
        src.distances[keep],
    )


def _match_exact_channel(ref: EventSet, cand: EventSet, *, time_tol: int) -> tuple[np.ndarray, np.ndarray]:
    ref_matches: list[np.ndarray] = []
    cand_matches: list[np.ndarray] = []
    for ch in np.intersect1d(np.unique(ref.channels), np.unique(cand.channels)):
        ri = np.flatnonzero(ref.channels == ch)
        ci = np.flatnonzero(cand.channels == ch)
        ro = ri[np.argsort(ref.times[ri], kind="stable")]
        co = ci[np.argsort(cand.times[ci], kind="stable")]
        i = 0
        j = 0
        rr: list[int] = []
        cc: list[int] = []
        while i < ro.size and j < co.size:
            rt = int(ref.times[ro[i]])
            ct = int(cand.times[co[j]])
            if ct < rt - time_tol:
                j += 1
            elif ct > rt + time_tol:
                i += 1
            else:
                rr.append(int(ro[i]))
                cc.append(int(co[j]))
                i += 1
                j += 1
        if rr:
            ref_matches.append(np.asarray(rr, dtype=np.int64))
            cand_matches.append(np.asarray(cc, dtype=np.int64))
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
    for ri in np.argsort(ref.times, kind="stable").tolist():
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


def _assignment_metrics(
    ref: EventSet,
    cand: EventSet,
    *,
    time_tol: int,
    channel_tol: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ri, ci = _match_with_channel_tol(ref, cand, time_tol=time_tol, channel_tol=channel_tol)
    matched = int(ri.size)
    ref_acc_all = ref.labels >= 0
    cand_acc_all = cand.labels >= 0
    ref_acc = ref.labels[ri] >= 0 if matched else np.zeros(0, dtype=bool)
    cand_acc = cand.labels[ci] >= 0 if matched else np.zeros(0, dtype=bool)
    both_acc = ref_acc & cand_acc
    label_same = np.zeros(ref_acc.shape, dtype=bool)
    if matched:
        label_same = ref.labels[ri] == cand.labels[ci]
    ref_accepted_matched = int(np.count_nonzero(ref_acc))
    label_correct = int(np.count_nonzero(label_same & ref_acc))
    ref_accepted_total = int(np.count_nonzero(ref_acc_all))
    cand_accepted_total = int(np.count_nonzero(cand_acc_all))
    label_recall = label_correct / ref_accepted_total if ref_accepted_total else None
    label_precision = label_correct / cand_accepted_total if cand_accepted_total else None
    label_f1 = None
    if label_recall is not None and label_precision is not None and (label_recall + label_precision) > 0:
        label_f1 = 2.0 * label_recall * label_precision / (label_recall + label_precision)
    summary = {
        "reference": ref.name,
        "candidate": cand.name,
        "reference_events": int(ref.times.size),
        "candidate_events": int(cand.times.size),
        "matched_time_channel": matched,
        "detection_recall": matched / int(ref.times.size) if ref.times.size else None,
        "detection_precision": matched / int(cand.times.size) if cand.times.size else None,
        "reference_accepted_total": ref_accepted_total,
        "candidate_accepted_total": cand_accepted_total,
        "reference_accepted_matched": ref_accepted_matched,
        "candidate_accepted_matched": int(np.count_nonzero(cand_acc)),
        "both_accepted_matched": int(np.count_nonzero(both_acc)),
        "false_reject_on_matched": int(np.count_nonzero(ref_acc & ~cand_acc)),
        "false_accept_on_matched": int(np.count_nonzero(~ref_acc & cand_acc)),
        "accept_status_match_ratio": float(np.mean(ref_acc == cand_acc)) if matched else None,
        "label_correct_on_reference_accepted_matched": label_correct,
        "label_acc_on_reference_accepted_matched": label_correct / ref_accepted_matched if ref_accepted_matched else None,
        "label_recall_on_reference_accepted_total": label_recall,
        "label_precision_on_candidate_accepted_total": label_precision,
        "label_f1_on_reference_candidate_accepted": label_f1,
        "label_e2e_on_reference_accepted_total": label_recall,
        "label_correct_on_both_accepted": int(np.count_nonzero(label_same & both_acc)),
        "label_acc_on_both_accepted": (
            int(np.count_nonzero(label_same & both_acc)) / int(np.count_nonzero(both_acc))
            if np.count_nonzero(both_acc)
            else None
        ),
    }

    mistake_rows: list[dict[str, Any]] = []
    wrong = ref_acc & cand_acc & ~label_same
    if np.any(wrong):
        pairs, counts = np.unique(
            np.column_stack([ref.labels[ri][wrong], cand.labels[ci][wrong]]),
            axis=0,
            return_counts=True,
        )
        order = np.argsort(counts)[::-1][:50]
        for idx in order.tolist():
            mistake_rows.append(
                {
                    "reference": ref.name,
                    "candidate": cand.name,
                    "reference_label": int(pairs[idx, 0]),
                    "candidate_label": int(pairs[idx, 1]),
                    "count": int(counts[idx]),
                }
            )
    return summary, mistake_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Neuron Assignment Comparison",
        "",
        f"reference_source: `{summary['reference_source']}`",
        f"time_tol_samples: `{summary['time_tol_samples']}`",
        f"channel_tol: `{summary['channel_tol']}`",
        f"common_range_samples: `{summary['common_range_samples']}`",
        "",
        "| candidate | matched | det recall | det precision | ref accepted matched | label acc matched | label recall/all ref accepted | label precision/all candidate accepted | label F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['candidate']} | {row['matched_time_channel']} | "
            f"{(row['detection_recall'] or 0.0):.4f} | {(row['detection_precision'] or 0.0):.4f} | "
            f"{row['reference_accepted_matched']} | "
            f"{(row['label_acc_on_reference_accepted_matched'] or 0.0):.4f} | "
            f"{(row['label_recall_on_reference_accepted_total'] or 0.0):.4f} | "
            f"{(row['label_precision_on_candidate_accepted_total'] or 0.0):.4f} | "
            f"{(row['label_f1_on_reference_candidate_accepted'] or 0.0):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    p.add_argument("--sw8-root", type=Path, default=DEFAULT_SW8_ROOT)
    p.add_argument("--hw8-root", type=Path, default=DEFAULT_HW8_ROOT)
    p.add_argument("--reference", choices=["full_precision", "sw8_software", "hw8_hardware"], default="sw8_software")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--chunk-len", type=int, default=300)
    p.add_argument("--chunk-start", type=int, default=None)
    p.add_argument("--chunk-stop", type=int, default=None)
    p.add_argument("--time-tol-samples", type=int, default=5)
    p.add_argument("--channel-tol", type=int, default=0)
    p.add_argument("--no-common-range", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sources = {
        "full_precision": _load_source("full_precision", args.full_root, chunk_len=args.chunk_len, chunk_start=args.chunk_start, chunk_stop=args.chunk_stop),
        "sw8_software": _load_source("sw8_software", args.sw8_root, chunk_len=args.chunk_len, chunk_start=args.chunk_start, chunk_stop=args.chunk_stop),
        "hw8_hardware": _load_source("hw8_hardware", args.hw8_root, chunk_len=args.chunk_len, chunk_start=args.chunk_start, chunk_stop=args.chunk_stop),
    }
    common_range: list[int | None] = [None, None]
    if not args.no_common_range:
        start = max(int(s.times.min()) for s in sources.values() if s.times.size)
        stop = min(int(s.times.max()) + 1 for s in sources.values() if s.times.size)
        common_range = [start, stop]
        sources = {k: _filter_time_range(v, start, stop) for k, v in sources.items()}

    ref = sources[args.reference]
    comparisons: list[dict[str, Any]] = []
    mistakes: list[dict[str, Any]] = []
    for name, cand in sources.items():
        if name == args.reference:
            continue
        row, rows = _assignment_metrics(ref, cand, time_tol=args.time_tol_samples, channel_tol=args.channel_tol)
        comparisons.append(row)
        mistakes.extend(rows)

    summary = {
        "format": "neuron_assignment_compare.v1",
        "reference_source": args.reference,
        "time_tol_samples": int(args.time_tol_samples),
        "channel_tol": int(args.channel_tol),
        "chunk_len": int(args.chunk_len),
        "common_range_samples": common_range,
        "sources": {
            name: {
                "root": src.root,
                "events": int(src.times.size),
                "accepted": int(np.count_nonzero(src.labels >= 0)),
                "min_sample": int(src.times.min()) if src.times.size else None,
                "max_sample": int(src.times.max()) if src.times.size else None,
            }
            for name, src in sources.items()
        },
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "neuron_assignment_summary.json", summary)
    _write_csv(
        args.output_dir / "neuron_assignment_comparisons.csv",
        comparisons,
        [
            "reference",
            "candidate",
            "matched_time_channel",
            "detection_recall",
            "detection_precision",
            "reference_accepted_total",
            "candidate_accepted_total",
            "reference_accepted_matched",
            "label_correct_on_reference_accepted_matched",
            "label_acc_on_reference_accepted_matched",
            "label_recall_on_reference_accepted_total",
            "label_precision_on_candidate_accepted_total",
            "label_f1_on_reference_candidate_accepted",
            "label_e2e_on_reference_accepted_total",
            "false_reject_on_matched",
            "false_accept_on_matched",
        ],
    )
    _write_csv(
        args.output_dir / "neuron_assignment_top_mistakes.csv",
        mistakes,
        ["reference", "candidate", "reference_label", "candidate_label", "count"],
    )
    _write_report(args.output_dir / "neuron_assignment_report.md", summary)
    print(f"summary -> {args.output_dir / 'neuron_assignment_summary.json'}")
    print(f"report  -> {args.output_dir / 'neuron_assignment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
