#!/usr/bin/env python3
"""Run clean-style choice decoding for IBL/full/8-bit software/8-bit hardware events."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CLEAN_ROOT = Path("/home/wyzhao/git/clean-20260611-214322")
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
    "/home/wyzhao/git/hardware_export/analysis/Code_for_analysis/results_choice_decoding"
)
TASK_NAME = "choice"


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


def _load_spikes(root: Path, *, sample_rate: float, time_offset_s: float) -> dict[str, np.ndarray | int]:
    time_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    channel_parts: list[np.ndarray] = []
    for path in _event_files(root):
        with np.load(path, allow_pickle=False) as z:
            times = np.asarray(z["times"], dtype=np.int64)
            labels = np.asarray(z["labels"], dtype=np.int32)
            channels = np.asarray(z["channels"], dtype=np.int32)
        keep = labels >= 0
        if np.any(keep):
            time_parts.append(times[keep])
            label_parts.append(labels[keep])
            channel_parts.append(channels[keep])
    if not time_parts:
        raise ValueError(f"no accepted/assigned spikes found under {root}")
    times_samples = np.concatenate(time_parts)
    labels = np.concatenate(label_parts)
    channels = np.concatenate(channel_parts)
    order = np.argsort(times_samples, kind="stable")
    return {
        "times": times_samples[order].astype(np.float64) / float(sample_rate) + float(time_offset_s),
        "times_samples": times_samples[order].astype(np.int64),
        "labels": labels[order].astype(np.int32),
        "channels": channels[order].astype(np.int32),
        "n_spikes": int(times_samples.size),
        "n_clusters": int(np.unique(labels).size),
        "max_time_s": float(times_samples.max() / float(sample_rate) + float(time_offset_s)),
    }


def _load_ibl_reference_spikes(
    *,
    one,
    config,
    load_channel_geometry,
    load_reference,
    sample_rate: float,
    spike_sorter: str,
) -> dict[str, Any]:
    positions, raw_ind = load_channel_geometry(one, config.IBL_SESSION_EID, config.IBL_PROBE_LABEL)
    _ = positions
    ref = load_reference(
        one,
        config.IBL_SESSION_EID,
        config.IBL_PROBE_LABEL,
        spike_sorter=spike_sorter,
        raw_ind=raw_ind,
    )
    if ref is None:
        raise RuntimeError("failed to load IBL reference spikes")
    times = np.asarray(ref["spike_times"], dtype=np.float64)
    labels = np.asarray(ref["spike_clusters"], dtype=np.int32)
    channels = np.asarray(ref["spike_channels"], dtype=np.int32)
    order = np.argsort(times, kind="stable")
    labels_ordered = labels[order]
    max_label = int(labels_ordered.max()) if labels_ordered.size else -1
    is_good = np.zeros(max_label + 1, dtype=bool)
    cluster_labels = np.asarray(ref.get("cluster_labels", []), dtype=np.int32)
    n = min(is_good.size, cluster_labels.size)
    if n:
        is_good[:n] = cluster_labels[:n] == 1
    return {
        "times": times[order],
        "times_samples": np.rint(times[order] * float(sample_rate)).astype(np.int64),
        "labels": labels_ordered,
        "channels": channels[order],
        "is_good": is_good,
        "quality_source": f"ibl_{ref.get('sorter', spike_sorter)}_cluster_label_eq_1",
        "n_spikes": int(times.size),
        "n_clusters": int(np.unique(labels).size),
        "max_time_s": float(times.max()) if times.size else 0.0,
    }


def _compute_firing_rates_fast(
    spike_times: np.ndarray,
    spike_labels: np.ndarray,
    cluster_ids: list[int],
    trial_times: np.ndarray,
    *,
    window: tuple[float, float],
) -> np.ndarray:
    pre, post = window
    duration = max(float(post - pre), 1e-6)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    spike_labels = np.asarray(spike_labels, dtype=np.int32)
    trial_times = np.asarray(trial_times, dtype=np.float64)
    cluster_ids_arr = np.asarray(cluster_ids, dtype=np.int32)
    rates = np.zeros((trial_times.size, cluster_ids_arr.size), dtype=np.float32)
    if trial_times.size == 0 or cluster_ids_arr.size == 0 or spike_times.size == 0:
        return rates

    max_label = int(max(int(spike_labels.max()), int(cluster_ids_arr.max())))
    label_to_col = np.full(max_label + 1, -1, dtype=np.int32)
    label_to_col[cluster_ids_arr] = np.arange(cluster_ids_arr.size, dtype=np.int32)
    starts = np.searchsorted(spike_times, trial_times + pre, side="left")
    ends = np.searchsorted(spike_times, trial_times + post, side="left")
    for i, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
        if hi <= lo:
            continue
        labels = spike_labels[lo:hi]
        valid = (labels >= 0) & (labels <= max_label)
        if not np.any(valid):
            continue
        cols = label_to_col[labels[valid]]
        cols = cols[cols >= 0]
        if cols.size:
            rates[i] = np.bincount(cols, minlength=cluster_ids_arr.size).astype(np.float32)
    rates /= duration
    return rates


def _run_decode_for_clusters(
    *,
    decode_task,
    compute_cluster_quality,
    trials: dict[str, Any],
    spike_data: dict[str, Any],
    cluster_mode: str,
    sample_rate: float,
    val_end_s: float,
    test_end_s: float | None,
    n_splits: int,
    n_shuffle: int,
    quant_bits: list[int],
) -> dict[str, Any]:
    choice_event_times, choice_event_label = decode_task.get_choice_event_times(trials)
    if choice_event_times is None:
        return {"skipped": True, "reason": "choice event unavailable"}
    valid = trials["valid"].copy()
    valid &= trials["stim_on"] >= val_end_s
    valid &= np.isfinite(trials["stim_on"])
    valid &= np.isfinite(trials["feedback_times"])
    valid &= np.isfinite(choice_event_times)
    if test_end_s is not None:
        valid &= trials["stim_on"] < test_end_s
        valid &= choice_event_times < test_end_s
    idx = np.where(valid)[0]
    y = (trials["choice"][idx] == 1).astype(np.int32)
    event_times = choice_event_times[idx]
    all_clusters = sorted(int(x) for x in np.unique(spike_data["labels"]) if int(x) >= 0)
    cluster_sets: dict[str, list[int]] = {}
    if cluster_mode in ("all", "both"):
        cluster_sets["all"] = all_clusters
    if cluster_mode in ("acg-good", "both"):
        if spike_data.get("is_good") is not None:
            is_good = np.asarray(spike_data["is_good"], dtype=bool)
        else:
            max_label = int(np.max(spike_data["labels"])) if spike_data["n_spikes"] else -1
            quality_input = {
                "times": spike_data["times_samples"],
                "labels": spike_data["labels"],
                "n_clusters_total": max_label + 1,
            }
            is_good, _contam = compute_cluster_quality(quality_input, sample_rate=sample_rate)
        active = set(all_clusters)
        cluster_sets["acg_good"] = [int(i) for i, ok in enumerate(is_good) if bool(ok) and int(i) in active]

    out: dict[str, Any] = {
        "task": TASK_NAME,
        "choice_event_label": choice_event_label,
        "trial_count_available": int(idx.size),
        "cluster_modes": {},
    }
    for mode, cluster_ids in cluster_sets.items():
        if idx.size < 30 or len(cluster_ids) < 3:
            out["cluster_modes"][mode] = {"skipped": True, "reason": "too few trials or clusters"}
            continue
        window = tuple(getattr(decode_task.config, "DECODE_CHOICE_WINDOW", (-0.1, 0.0)))
        x = _compute_firing_rates_fast(
            spike_data["times"], spike_data["labels"], cluster_ids, event_times, window=window
        )
        active = x.std(axis=0) > 1e-6
        if int(np.count_nonzero(active)) < 2:
            out["cluster_modes"][mode] = {"skipped": True, "reason": "too few active clusters"}
            continue
        res = decode_task.decode_logistic(
            x[:, active],
            y,
            n_splits=n_splits,
            n_shuffle=n_shuffle,
            quant_bits=quant_bits,
        )
        out["cluster_modes"][mode] = {
            "skipped": False,
            "balanced_accuracy": float(res["real_acc"]),
            "null_mean": float(res["null_mean"]),
            "null_std": float(res["null_std"]),
            "p_value": float(res["p_value"]),
            "n_trials": int(res["n_trials"]),
            "n_features": int(res["n_features"]),
            "n_nonzero": int(res.get("n_nonzero", 0)),
            "quant_accs": {str(k): float(v) for k, v in res.get("quant_accs", {}).items()},
        }
    return out


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Choice Decoding",
        "",
        f"calibration_duration_s: `{summary['calibration_duration_s']}`",
        f"validation_duration_s: `{summary['validation_duration_s']}`",
        f"test_end_s: `{summary['test_end_s']}`",
        "",
        "| source | cluster mode | spikes | clusters | bal acc | null | p | trials | features |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for src in summary["sources"]:
        for mode, row in src["decode"]["cluster_modes"].items():
            if row.get("skipped"):
                lines.append(f"| {src['name']} | {mode} | {src['n_spikes']} | {src['n_clusters']} | skip | | | | |")
            else:
                lines.append(
                    f"| {src['name']} | {mode} | {src['n_spikes']} | {src['n_clusters']} | "
                    f"{row['balanced_accuracy']:.4f} | {row['null_mean']:.4f}+/-{row['null_std']:.4f} | "
                    f"{row['p_value']:.4g} | {row['n_trials']} | {row['n_features']} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    p.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    p.add_argument("--sw8-root", type=Path, default=DEFAULT_SW8_ROOT)
    p.add_argument("--hw8-root", type=Path, default=DEFAULT_HW8_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--source",
        action="append",
        choices=["ibl_reference", "full_precision", "sw8_software", "hw8_hardware"],
        default=None,
    )
    p.add_argument("--sample-rate", type=float, default=30000.0)
    p.add_argument("--ibl-spike-sorter", default="iblsorter")
    p.add_argument("--time-offset-s", type=float, default=0.0)
    p.add_argument("--calibration-duration-s", type=float, default=240.0)
    p.add_argument("--validation-duration-s", type=float, default=120.0)
    p.add_argument("--test-end-s", type=float, default=None)
    p.add_argument("--no-common-test-end", action="store_true")
    p.add_argument("--cluster-mode", choices=["all", "acg-good", "both"], default="both")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-shuffle", type=int, default=200)
    p.add_argument("--quant-bits", default="2,4,6,8")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*")
    sys.path.insert(0, str(args.clean_root))
    import decode_task  # noqa: PLC0415
    import config  # noqa: PLC0415
    from data_loader import connect_one, load_channel_geometry  # noqa: PLC0415
    from merging import compute_cluster_quality  # noqa: PLC0415
    from comparison import load_reference  # noqa: PLC0415

    source_roots = {
        "ibl_reference": None,
        "full_precision": args.full_root,
        "sw8_software": args.sw8_root,
        "hw8_hardware": args.hw8_root,
    }
    selected = args.source or ["ibl_reference", "full_precision", "sw8_software", "hw8_hardware"]
    quant_bits = [int(x) for x in args.quant_bits.split(",") if x.strip()]
    one = connect_one(silent=True)
    loaded = []
    for name in selected:
        if name == "ibl_reference":
            spikes = _load_ibl_reference_spikes(
                one=one,
                config=config,
                load_channel_geometry=load_channel_geometry,
                load_reference=load_reference,
                sample_rate=args.sample_rate,
                spike_sorter=args.ibl_spike_sorter,
            )
        else:
            spikes = _load_spikes(source_roots[name], sample_rate=args.sample_rate, time_offset_s=args.time_offset_s)
        loaded.append((name, source_roots[name], spikes))
    test_end_s = args.test_end_s
    if test_end_s is None and not args.no_common_test_end:
        test_end_s = min(float(spikes["max_time_s"]) for _name, _root, spikes in loaded)

    trials = decode_task.load_trials(one, config.IBL_SESSION_EID)
    val_end_s = float(args.calibration_duration_s) + float(args.validation_duration_s)
    sources = []
    for name, root, spikes in loaded:
        dec = _run_decode_for_clusters(
            decode_task=decode_task,
            compute_cluster_quality=compute_cluster_quality,
            trials=trials,
            spike_data=spikes,
            cluster_mode=args.cluster_mode,
            sample_rate=args.sample_rate,
            val_end_s=val_end_s,
            test_end_s=test_end_s,
            n_splits=args.n_splits,
            n_shuffle=args.n_shuffle,
            quant_bits=quant_bits,
        )
        sources.append(
            {
                "name": name,
                "root": root,
                "n_spikes": int(spikes["n_spikes"]),
                "n_clusters": int(spikes["n_clusters"]),
                "max_time_s": float(spikes["max_time_s"]),
                "quality_source": spikes.get("quality_source", "acg_computed"),
                "decode": dec,
            }
        )
    summary = {
        "format": "choice_decoding_compare.v1",
        "clean_root": args.clean_root,
        "sample_rate": float(args.sample_rate),
        "time_offset_s": float(args.time_offset_s),
        "calibration_duration_s": float(args.calibration_duration_s),
        "validation_duration_s": float(args.validation_duration_s),
        "test_end_s": test_end_s,
        "cluster_mode": args.cluster_mode,
        "n_splits": int(args.n_splits),
        "n_shuffle": int(args.n_shuffle),
        "quant_bits": quant_bits,
        "sources": sources,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "choice_decoding_summary.json", summary)
    _write_report(args.output_dir / "choice_decoding_report.md", summary)
    print(f"summary -> {args.output_dir / 'choice_decoding_summary.json'}")
    print(f"report  -> {args.output_dir / 'choice_decoding_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
