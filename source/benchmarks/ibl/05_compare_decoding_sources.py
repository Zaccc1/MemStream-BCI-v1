#!/usr/bin/env python3
"""Compare decoding against behavioral labels for IBL/full/sw8/hardware events."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CLEAN_ROOT = Path("/home/wyzhao/git/clean-20260611-214322")
DEFAULT_IBL_SESSION_ROOT = Path(
    "/home/wyzhao/Downloads/ONE/openalyx.internationalbrainlab.org/mainenlab/"
    "Subjects/ZFM-01936/2021-01-22/002"
)
DEFAULT_FULL_ROOT = Path(
    "/home/wyzhao/git/hardware_export/recordings/golden_full_precision/"
    "full_precision_tail3_sharded_0.000s_4266545ms"
)
DEFAULT_SW8_ROOT = Path(
    "/home/wyzhao/git/hardware_export/analysis/vmm/"
    "tail3_full_recording_353_topk10_cal5s_streamall_20260617_215844/"
    "golden/clean_recording_tail3_sharded_0.000s_4266545ms"
)
DEFAULT_HW_ROOT = Path(
    "/home/wyzhao/git/hardware_export/analysis/vmm/"
    "(U)full_recording_hybrid_old313_backend_spi20_20260620_163453/"
    "actual_events"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/wyzhao/git/hardware_export/analysis/Code_for_analysis/"
    "results_decoding_sources_behavior_benchmark"
)


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


def _parse_start_s(root: Path) -> float:
    m = re.search(r"tail3_sharded_([0-9]+(?:\.[0-9]+)?)s_", root.name)
    if m:
        return float(m.group(1))
    m = re.search(r"tail3_([0-9]+(?:\.[0-9]+)?)s_", root.name)
    if m:
        return float(m.group(1))
    return 0.0


def _chunk_start_from_name(path: Path) -> int:
    m = re.search(r"chunks_(\d+)_", path.name)
    return int(m.group(1)) if m else 0


def _event_files(root: Path) -> list[Path]:
    manifest = root / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        files: list[Path] = []
        for item in data.get("shards", []):
            p = item.get("file") or item.get("actual_events_file")
            if p:
                pp = Path(p)
                candidate = pp if pp.is_absolute() else root / pp
                if not candidate.exists():
                    fallback = root / pp.name
                    if fallback.exists():
                        candidate = fallback
                files.append(candidate)
        if files:
            return files
    if (root / "tail3_shards").is_dir():
        root = root / "tail3_shards"
    if (root / "actual_events").is_dir():
        root = root / "actual_events"
    files = sorted(root.glob("shard_*_chunks_*.npz"), key=_chunk_start_from_name)
    files += sorted(root.glob("actual_shard_*_chunks_*.npz"), key=_chunk_start_from_name)
    if not files:
        raise FileNotFoundError(f"no event shards under {root}")
    return files


def _load_sharded_events(root: Path, *, sample_rate: float, time_offset_s: float) -> dict[str, Any]:
    time_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    channel_parts: list[np.ndarray] = []
    for path in _event_files(root):
        with np.load(path, mmap_mode="r") as z:
            times = np.asarray(z["times"], dtype=np.int64)
            labels = np.asarray(z["labels"], dtype=np.int32)
            channels = np.asarray(z["channels"], dtype=np.int32)
        if not (times.shape == labels.shape == channels.shape):
            raise ValueError(f"shape mismatch in {path}: {times.shape} {labels.shape} {channels.shape}")
        keep = labels >= 0
        if np.any(keep):
            time_parts.append(times[keep])
            label_parts.append(labels[keep])
            channel_parts.append(channels[keep])
    if not time_parts:
        raise ValueError(f"no accepted/assigned events under {root}")
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
        "quality_source": "acg_computed",
    }


def _find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no match for {pattern} under {path}")
    return matches[-1]


def _load_trials_local(session_root: Path) -> dict[str, Any]:
    table_path = _find_one(session_root / "alf", "**/_ibl_trials.table.pqt")
    df = pd.read_parquet(table_path)
    n = len(df)

    def col(name: str, default: float = np.nan) -> np.ndarray:
        if name in df:
            return df[name].to_numpy()
        return np.full(n, default, dtype=np.float64)

    choice = col("choice")
    feedback = col("feedbackType")
    stim_on = col("stimOn_times")
    feedback_t = col("feedback_times")
    first_movement = col("firstMovement_times")
    go_cue = col("goCue_times")
    valid = (
        np.isfinite(stim_on)
        & np.isfinite(feedback_t)
        & np.isin(choice, [-1, 1])
        & np.isin(feedback, [-1, 1])
    )
    return {
        "choice": choice,
        "feedback": feedback,
        "stim_on": stim_on,
        "feedback_times": feedback_t,
        "first_movement": first_movement,
        "go_cue": go_cue,
        "interval_start": col("intervals_0"),
        "interval_stop": col("intervals_1"),
        "contrast_left": col("contrastLeft"),
        "contrast_right": col("contrastRight"),
        "valid": valid,
        "n_trials": n,
        "source_file": table_path,
    }


def _load_ibl_reference(session_root: Path, *, sample_rate: float) -> dict[str, Any]:
    base = _find_one(session_root / "alf" / "probe00", "**/spikes.times.npy").parent
    spike_times = np.load(base / "spikes.times.npy", mmap_mode="r")
    spike_labels = np.load(base / "spikes.clusters.npy", mmap_mode="r").astype(np.int32)
    cluster_channels = np.load(base / "clusters.channels.npy", mmap_mode="r").astype(np.int32)
    raw_ind_path = base / "channels.rawInd.npy"
    if raw_ind_path.exists():
        raw_ind = np.load(raw_ind_path, mmap_mode="r").astype(np.int32)
        valid = (cluster_channels >= 0) & (cluster_channels < raw_ind.size)
        cluster_channels_raw = np.zeros_like(cluster_channels)
        cluster_channels_raw[valid] = raw_ind[cluster_channels[valid]]
        cluster_channels = cluster_channels_raw
    channels = cluster_channels[spike_labels]

    metrics = pd.read_parquet(base / "clusters.metrics.pqt")
    max_label = int(spike_labels.max()) if spike_labels.size else -1
    is_good = np.zeros(max_label + 1, dtype=bool)
    if "label" in metrics:
        for _, row in metrics.iterrows():
            cid = int(row["cluster_id"]) if "cluster_id" in metrics else int(row.name)
            if 0 <= cid < is_good.size:
                val = row["label"]
                is_good[cid] = bool(np.isfinite(val) and float(val) > 0.8)
        quality_source = "ibl_metrics_label_gt_0p8"
    elif "ks2_label" in metrics:
        for _, row in metrics.iterrows():
            cid = int(row["cluster_id"]) if "cluster_id" in metrics else int(row.name)
            if 0 <= cid < is_good.size:
                is_good[cid] = str(row["ks2_label"]).lower() == "good"
        quality_source = "ibl_metrics_ks2_label_good"
    else:
        quality_source = "none"

    order = np.argsort(spike_times, kind="stable")
    times = np.asarray(spike_times[order], dtype=np.float64)
    labels = np.asarray(spike_labels[order], dtype=np.int32)
    return {
        "times": times,
        "times_samples": np.rint(times * float(sample_rate)).astype(np.int64),
        "labels": labels,
        "channels": np.asarray(channels[order], dtype=np.int32),
        "n_spikes": int(labels.size),
        "n_clusters": int(np.unique(labels).size),
        "is_good": is_good,
        "quality_source": quality_source,
        "source_dir": base,
    }


def _choice_event_times(trials: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    for label, key in (
        ("first movement", "first_movement"),
        ("choice event", "choice_times"),
        ("go cue", "go_cue"),
        ("stimulus onset", "stim_on"),
    ):
        arr = trials.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size and np.isfinite(arr).any():
            return arr, label
    return None, None


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


def _cluster_sets(spikes: dict[str, Any], compute_cluster_quality, sample_rate: float) -> dict[str, list[int]]:
    all_clusters = sorted(int(x) for x in np.unique(spikes["labels"]) if int(x) >= 0)
    out = {"all": all_clusters}
    if spikes.get("is_good") is not None:
        is_good = np.asarray(spikes["is_good"], dtype=bool)
    else:
        max_label = int(np.max(spikes["labels"])) if spikes["n_spikes"] else -1
        quality_input = {
            "times": spikes["times_samples"],
            "labels": spikes["labels"],
            "n_clusters_total": max_label + 1,
        }
        is_good, _contam = compute_cluster_quality(quality_input, sample_rate=sample_rate)
    active = set(all_clusters)
    out["good"] = [int(i) for i, ok in enumerate(is_good) if bool(ok) and int(i) in active]
    return out


def _run_decode(
    *,
    decode_task,
    trials: dict[str, Any],
    spikes: dict[str, Any],
    cluster_ids: list[int],
    val_end_s: float,
    test_end_s: float | None,
    n_splits: int,
    n_shuffle: int,
    quant_bits: list[int],
) -> dict[str, Any]:
    valid_base = trials["valid"].copy()
    valid_base &= trials["stim_on"] >= val_end_s
    valid_base &= np.isfinite(trials["stim_on"])
    valid_base &= np.isfinite(trials["feedback_times"])
    if test_end_s is not None:
        valid_base &= trials["stim_on"] < test_end_s
        valid_base &= trials["feedback_times"] < test_end_s

    choice_times, choice_event_label = _choice_event_times(trials)
    tasks: dict[str, Any] = {"choice_event_label": choice_event_label}

    def one_task(task: str, idx: np.ndarray, event_times: np.ndarray, y: np.ndarray, window: tuple[float, float]) -> None:
        if idx.size < 30 or len(cluster_ids) < 3:
            tasks[task] = {"skipped": True, "reason": "too few trials or clusters"}
            return
        x = _compute_firing_rates_fast(spikes["times"], spikes["labels"], cluster_ids, event_times, window=window)
        active = x.std(axis=0) > 1e-6
        if int(np.count_nonzero(active)) < 2:
            tasks[task] = {"skipped": True, "reason": "too few active clusters"}
            return
        res = decode_task.decode_logistic(
            x[:, active],
            y.astype(np.int32),
            n_splits=n_splits,
            n_shuffle=n_shuffle,
            quant_bits=quant_bits,
        )
        tasks[task] = {
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

    if choice_times is not None:
        choice_valid = valid_base & np.isfinite(choice_times)
        if test_end_s is not None:
            choice_valid &= choice_times < test_end_s
        idx = np.where(choice_valid)[0]
        one_task(
            "choice",
            idx,
            choice_times[idx],
            (trials["choice"][idx] == 1).astype(np.int32),
            tuple(getattr(decode_task.config, "DECODE_CHOICE_WINDOW", (-0.1, 0.0))),
        )
    else:
        tasks["choice"] = {"skipped": True, "reason": "choice event unavailable"}

    idx = np.where(valid_base)[0]
    one_task(
        "feedback",
        idx,
        trials["feedback_times"][idx],
        (trials["feedback"][idx] == 1).astype(np.int32),
        tuple(getattr(decode_task.config, "DECODE_FEEDBACK_WINDOW", (0.0, 0.2))),
    )
    return tasks


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Decoding Sources vs Behavior",
        "",
        f"trial_source: `{summary['trial_source']}`",
        f"calibration_duration_s: `{summary['calibration_duration_s']}`",
        f"validation_duration_s: `{summary['validation_duration_s']}`",
        f"test_end_s: `{summary['test_end_s']}`",
        "",
        "| source | cluster set | spikes | clusters | task | bal acc | null | p | trials | features |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for src in summary["sources"]:
        for cluster_set, cluster_row in src["cluster_sets"].items():
            for task in ("choice", "feedback"):
                row = cluster_row["tasks"][task]
                if row.get("skipped"):
                    lines.append(
                        f"| {src['name']} | {cluster_set} | {src['n_spikes']} | "
                        f"{cluster_row['n_clusters']} | {task} | skip | | | | |"
                    )
                    continue
                lines.append(
                    f"| {src['name']} | {cluster_set} | {src['n_spikes']} | "
                    f"{cluster_row['n_clusters']} | {task} | "
                    f"{row['balanced_accuracy']:.4f} | "
                    f"{row['null_mean']:.4f}+/-{row['null_std']:.4f} | "
                    f"{row['p_value']:.4g} | {row['n_trials']} | {row['n_features']} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    p.add_argument("--ibl-session-root", type=Path, default=DEFAULT_IBL_SESSION_ROOT)
    p.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    p.add_argument("--sw8-root", type=Path, default=DEFAULT_SW8_ROOT)
    p.add_argument("--hw-root", type=Path, default=DEFAULT_HW_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--source",
        action="append",
        choices=["ibl_reference", "sw_full_precision", "sw_8bits", "hardware"],
        default=None,
    )
    p.add_argument("--sample-rate", type=float, default=30000.0)
    p.add_argument("--calibration-duration-s", type=float, default=240.0)
    p.add_argument("--validation-duration-s", type=float, default=120.0)
    p.add_argument("--test-end-s", type=float, default=4266.54)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-shuffle", type=int, default=200)
    p.add_argument("--quant-bits", default="2,4,6,8")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*")
    sys.path.insert(0, str(args.clean_root))
    import decode_task  # noqa: PLC0415
    from merging import compute_cluster_quality  # noqa: PLC0415

    selected = args.source or ["ibl_reference", "sw_full_precision", "sw_8bits", "hardware"]
    quant_bits = [int(x) for x in args.quant_bits.split(",") if x.strip()]
    val_end_s = float(args.calibration_duration_s) + float(args.validation_duration_s)
    trials = _load_trials_local(args.ibl_session_root)
    source_specs = {
        "ibl_reference": ("ibl_reference", args.ibl_session_root),
        "sw_full_precision": ("sw_full_precision", args.full_root),
        "sw_8bits": ("sw_8bits", args.sw8_root),
        "hardware": ("hardware", args.hw_root),
    }
    sources: list[dict[str, Any]] = []
    for name in selected:
        if name == "ibl_reference":
            spikes = _load_ibl_reference(args.ibl_session_root, sample_rate=args.sample_rate)
        else:
            root = source_specs[name][1]
            spikes = _load_sharded_events(
                root,
                sample_rate=args.sample_rate,
                time_offset_s=_parse_start_s(root),
            )
        sets = _cluster_sets(spikes, compute_cluster_quality, args.sample_rate)
        cluster_sets: dict[str, Any] = {}
        for set_name, cluster_ids in sets.items():
            tasks = _run_decode(
                decode_task=decode_task,
                trials=trials,
                spikes=spikes,
                cluster_ids=cluster_ids,
                val_end_s=val_end_s,
                test_end_s=args.test_end_s,
                n_splits=args.n_splits,
                n_shuffle=args.n_shuffle,
                quant_bits=quant_bits,
            )
            cluster_sets[set_name] = {
                "n_clusters": int(len(cluster_ids)),
                "tasks": tasks,
            }
        sources.append(
            {
                "name": source_specs[name][0],
                "root": source_specs[name][1],
                "n_spikes": int(spikes["n_spikes"]),
                "n_clusters_total": int(spikes["n_clusters"]),
                "quality_source": spikes.get("quality_source"),
                "cluster_sets": cluster_sets,
            }
        )

    summary = {
        "format": "decoding_sources_behavior_benchmark.v1",
        "clean_root": args.clean_root,
        "trial_source": trials["source_file"],
        "sample_rate": float(args.sample_rate),
        "calibration_duration_s": float(args.calibration_duration_s),
        "validation_duration_s": float(args.validation_duration_s),
        "test_end_s": float(args.test_end_s) if args.test_end_s is not None else None,
        "n_splits": int(args.n_splits),
        "n_shuffle": int(args.n_shuffle),
        "quant_bits": quant_bits,
        "sources": sources,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "decoding_sources_summary.json", summary)
    _write_report(args.output_dir / "decoding_sources_report.md", summary)
    print(f"summary -> {args.output_dir / 'decoding_sources_summary.json'}")
    print(f"report  -> {args.output_dir / 'decoding_sources_report.md'}")
    for src in sources:
        for cluster_set, cluster_row in src["cluster_sets"].items():
            for task in ("choice", "feedback"):
                row = cluster_row["tasks"][task]
                if row.get("skipped"):
                    print(f"{src['name']} {cluster_set} {task}: skipped")
                else:
                    print(
                        f"{src['name']} {cluster_set} {task}: "
                        f"acc={row['balanced_accuracy']:.4f} "
                        f"trials={row['n_trials']} features={row['n_features']}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
