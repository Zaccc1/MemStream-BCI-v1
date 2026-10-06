#!/usr/bin/env python3
"""Summarize one recording benchmark from existing pipeline outputs.

This script is intentionally a reporting layer.  It reuses the project's
existing decoding and comparison semantics:

- choice/feedback use decode_task.py feature extraction and logistic decoder;
- wheel movement uses the same paper-style lasso helper as
  09_decode_wheel_movement.py;
- KS4 precision/recall are read from comparison_vs_ks4_full, which is produced
  by the main pipeline through comparison.py / ks4_full_reference.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


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
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _parse_value(text: str) -> Any:
    text = text.strip()
    if text in {"True", "False"}:
        return text == "True"
    try:
        if any(c in text for c in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _read_summary_txt(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = _parse_value(value)
    return out


def _load_main_decoding_summary(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "decoding_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_ks4_cache(*roots: Path) -> Path | None:
    for root in roots:
        for path in sorted((root / "ks4_cache_full").glob("ks4_full_*s_*ch.npz")):
            return path
    return None


def _load_pipeline_clust_results(results_dir: Path) -> dict[str, np.ndarray]:
    for rel in ("clustering", "version_B/clustering"):
        d = results_dir / rel
        if (d / "spike_times.npy").exists():
            return {
                "times": np.load(d / "spike_times.npy", allow_pickle=False),
                "labels": np.load(d / "spike_clusters.npy", allow_pickle=False),
                "channels": np.load(d / "spike_channels.npy", allow_pickle=False),
            }
    raise FileNotFoundError(f"no clustering results found under {results_dir}")


def _load_ks4_full_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        out = {
            "all_times_raw": np.asarray(z["all_times_raw"], dtype=np.int64),
            "all_channels": np.asarray(z["all_channels"], dtype=np.int32),
            "all_labels": np.asarray(z["all_labels"], dtype=np.int32),
        }
        if "is_ref" in z:
            out["is_ref"] = np.asarray(z["is_ref"], dtype=bool)
    return out

def _valid_spike_events(
    times_s: np.ndarray,
    channels: np.ndarray,
    *,
    positions: np.ndarray,
    labels: np.ndarray | None = None,
    assigned_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    times_s = np.asarray(times_s, dtype=np.float64)
    channels = np.asarray(channels, dtype=np.int32)
    keep = np.isfinite(times_s)
    keep &= channels >= 0
    keep &= channels < int(positions.shape[0])
    if assigned_only and labels is not None:
        keep &= np.asarray(labels, dtype=np.int32) >= 0
    times_s = times_s[keep]
    channels = channels[keep]
    order = np.argsort(times_s, kind="stable")
    return times_s[order], channels[order]


def _one_to_one_event_metrics(
    *,
    pipe_times_s: np.ndarray,
    pipe_channels: np.ndarray,
    ref_times_s: np.ndarray,
    ref_channels: np.ndarray,
    positions: np.ndarray | None,
    max_dt_ms: float = 0.5,
    max_dist_um: float | None = 100.0,
) -> dict[str, Any]:
    from comparison import _count_coincidences

    use_spatial = positions is not None and max_dist_um is not None
    if pipe_times_s.size == 0 or ref_times_s.size == 0:
        matched = 0
    else:
        matched = int(_count_coincidences(
            pipe_times_s,
            ref_times_s,
            max_dt_ms / 1000.0,
            ch1=pipe_channels if use_spatial else None,
            ch2=ref_channels if use_spatial else None,
            positions=positions if use_spatial else None,
            max_dist=float(max_dist_um) if use_spatial else 100.0,
        ))
    return {
        "matched": matched,
        "pipe_events": int(pipe_times_s.size),
        "ref_events": int(ref_times_s.size),
        "precision": float(matched / max(int(pipe_times_s.size), 1)),
        "recall": float(matched / max(int(ref_times_s.size), 1)),
        "max_dt_ms": float(max_dt_ms),
        "max_dist_um": None if max_dist_um is None else float(max_dist_um),
        "spatial": bool(use_spatial),
    }


def _ks4_event_detection_metrics(
    *,
    spikes: dict[str, Any],
    ks4_cache: Path,
    positions: np.ndarray,
) -> dict[str, Any]:
    """Layer-1 event metrics: no cluster identity, only time/space events."""
    ks4_full = _load_ks4_full_cache(ks4_cache)
    sr = 30_000.0
    try:
        import config
        sr = float(getattr(config, "SAMPLE_RATE", sr))
    except Exception:
        pass

    pipe_times_all, pipe_ch_all = _valid_spike_events(
        spikes["times"],
        spikes["channels"],
        positions=positions,
        labels=spikes.get("labels"),
        assigned_only=False,
    )
    pipe_times_assigned, pipe_ch_assigned = _valid_spike_events(
        spikes["times"],
        spikes["channels"],
        positions=positions,
        labels=spikes.get("labels"),
        assigned_only=True,
    )
    if pipe_times_all.size:
        t_start = float(pipe_times_all.min())
        t_end = float(pipe_times_all.max())
    else:
        t_start = 0.0
        t_end = 0.0

    ref_times_abs = np.asarray(ks4_full["all_times_raw"], dtype=np.float64) / sr
    ref_channels_abs = np.asarray(ks4_full["all_channels"], dtype=np.int32)
    ref_labels_abs = np.asarray(ks4_full["all_labels"], dtype=np.int32)
    ref_mask = (ref_times_abs >= t_start) & (ref_times_abs <= t_end)
    ref_times_all, ref_ch_all = _valid_spike_events(
        ref_times_abs[ref_mask],
        ref_channels_abs[ref_mask],
        positions=positions,
    )

    good_ref_mask = np.zeros(np.count_nonzero(ref_mask), dtype=bool)
    is_ref = ks4_full.get("is_ref")
    if is_ref is not None:
        labels_w = ref_labels_abs[ref_mask]
        valid = (labels_w >= 0) & (labels_w < len(is_ref))
        good_ref_mask[valid] = np.asarray(is_ref, dtype=bool)[labels_w[valid]]
    ref_times_good, ref_ch_good = _valid_spike_events(
        ref_times_abs[ref_mask][good_ref_mask],
        ref_channels_abs[ref_mask][good_ref_mask],
        positions=positions,
    )

    return {
        "time_window_s": [t_start, t_end],
        "all_events_all_pipeline_spikes": _one_to_one_event_metrics(
            pipe_times_s=pipe_times_all,
            pipe_channels=pipe_ch_all,
            ref_times_s=ref_times_all,
            ref_channels=ref_ch_all,
            positions=positions,
        ),
        "good_events_all_pipeline_spikes": _one_to_one_event_metrics(
            pipe_times_s=pipe_times_all,
            pipe_channels=pipe_ch_all,
            ref_times_s=ref_times_good,
            ref_channels=ref_ch_good,
            positions=positions,
        ),
        "all_events_assigned_pipeline_spikes": _one_to_one_event_metrics(
            pipe_times_s=pipe_times_assigned,
            pipe_channels=pipe_ch_assigned,
            ref_times_s=ref_times_all,
            ref_channels=ref_ch_all,
            positions=positions,
        ),
        "good_events_assigned_pipeline_spikes": _one_to_one_event_metrics(
            pipe_times_s=pipe_times_assigned,
            pipe_channels=pipe_ch_assigned,
            ref_times_s=ref_times_good,
            ref_channels=ref_ch_good,
            positions=positions,
        ),
        "time_only_all_events_all_pipeline_spikes": _one_to_one_event_metrics(
            pipe_times_s=pipe_times_all,
            pipe_channels=pipe_ch_all,
            ref_times_s=ref_times_all,
            ref_channels=ref_ch_all,
            positions=None,
            max_dist_um=None,
        ),
        "time_only_all_events_assigned_pipeline_spikes": _one_to_one_event_metrics(
            pipe_times_s=pipe_times_assigned,
            pipe_channels=pipe_ch_assigned,
            ref_times_s=ref_times_all,
            ref_channels=ref_ch_all,
            positions=None,
            max_dist_um=None,
        ),
    }

def _compute_ks4_good_metrics(
    *,
    results_dir: Path,
    ks4_cache: Path,
    positions: np.ndarray,
) -> dict[str, Any]:
    from comparison import match_units
    from ks4_full_reference import _build_ks4_full_ref_dict, _get_pipeline_time_window

    clust_results = _load_pipeline_clust_results(results_dir)
    ks4_full = _load_ks4_full_cache(ks4_cache)
    t_start, t_end = _get_pipeline_time_window(clust_results)
    ref_filtered = _build_ks4_full_ref_dict(ks4_full, t_start, t_end)
    matches, summary = match_units(
        clust_results,
        ref_filtered,
        channel_positions=positions,
    )
    summary["computed_from_shared_ks4_cache"] = str(ks4_cache)
    summary["n_matches"] = len(matches)
    return summary


def _has_strict_ks4_metrics(summary: dict[str, Any]) -> bool:
    """Return True only when the summary contains real strict matching metrics."""
    return any(
        key in summary
        for key in (
            "n_matched_pairs",
            "n_ref_good",
            "good_TP",
            "good_global_recall",
            "global_recall",
        )
    )


def _augment_good_match_thresholds(summary: dict[str, Any], matches: np.ndarray) -> None:
    good = matches[matches["ref_is_good"]] if matches.size else matches
    if good.size:
        good_tp = int(np.sum(good["n_matched"]))
        good_pipe_total = int(np.sum(good["n_pipe"]))
        good_ref_total = int(np.sum(good["n_ref"]))
        summary.setdefault("good_TP", good_tp)
        summary.setdefault("good_pipe_total", good_pipe_total)
        summary.setdefault("good_ref_total", good_ref_total)
        summary.setdefault("good_global_precision", good_tp / max(good_pipe_total, 1))
        summary.setdefault("good_global_recall", good_tp / max(good_ref_total, 1))
        summary.setdefault("mean_precision_good", float(np.mean(good["precision"])))
        summary.setdefault("mean_recall_good", float(np.mean(good["recall"])))
        n_ref_good = max(int(summary.get("n_ref_good", 0)), 1)
        summary.setdefault("good_unit_coverage_any", float(good.size / n_ref_good))
        for threshold in (0.1, 0.2, 0.5):
            key = f"good_unit_coverage_recall_ge_{str(threshold).replace('.', 'p')}"
            summary.setdefault(
                key,
                float(np.count_nonzero(good["recall"] >= threshold) / n_ref_good),
            )
        for threshold in (0.5, 0.8):
            key = f"good_unit_coverage_accuracy_ge_{str(threshold).replace('.', 'p')}"
            summary.setdefault(
                key,
                float(np.count_nonzero(good["accuracy"] >= threshold) / n_ref_good),
            )
    summary["n_good_matched_pairs_from_file"] = int(good.size)


def _ks4_good_metrics(
    results_dir: Path,
    *,
    ks4_cache: Path | None,
    positions: np.ndarray | None,
) -> dict[str, Any]:
    comp_dir = results_dir / "comparison_vs_ks4_full"
    summary = _read_summary_txt(comp_dir / "comparison_summary.txt")
    matches_path = comp_dir / "unit_matches.npy"
    if matches_path.exists():
        matches = np.load(matches_path, allow_pickle=False)
        _augment_good_match_thresholds(summary, matches)
        summary["comparison_dir"] = comp_dir
        summary["strict_metric_source"] = "comparison_vs_ks4_full"

    if not _has_strict_ks4_metrics(summary) and ks4_cache is not None and positions is not None:
        summary = _compute_ks4_good_metrics(
            results_dir=results_dir,
            ks4_cache=ks4_cache,
            positions=positions,
        )
        summary["comparison_dir"] = comp_dir
        summary["strict_metric_source"] = "computed_from_shared_ks4_cache"
    else:
        summary.setdefault("comparison_dir", comp_dir)

    n_ref_good = int(summary.get("n_ref_good", 0))
    n_good_matched = int(summary.get("n_good_matched", 0))
    if n_ref_good > 0:
        summary.setdefault("good_unit_coverage_any", float(n_good_matched / n_ref_good))
    return summary


def _cluster_ids(spikes: dict[str, Any], mode: str) -> list[int]:
    labels = np.asarray(spikes["labels"], dtype=np.int32)
    active = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)
    if mode == "all":
        return active
    is_good = spikes.get("is_good")
    if is_good is None:
        return active
    is_good = np.asarray(is_good, dtype=bool)
    active_set = set(active)
    return [int(i) for i, ok in enumerate(is_good) if bool(ok) and int(i) in active_set]


def _decode_task_result(
    *,
    decode_task,
    spikes: dict[str, Any],
    cluster_ids: list[int],
    trials: dict[str, Any],
    task: str,
    val_end_s: float,
    test_end_s: float | None,
    n_splits: int,
    n_shuffle: int,
    quant_bits: list[int],
) -> dict[str, Any]:
    valid = trials["valid"].copy()
    valid &= trials["stim_on"] >= val_end_s
    valid &= np.isfinite(trials["stim_on"])
    valid &= np.isfinite(trials["feedback_times"])

    if task == "choice":
        event_times, align_label = decode_task.get_choice_event_times(trials)
        if event_times is None:
            return {"skipped": True, "reason": "choice event unavailable"}
        valid &= np.isfinite(event_times)
        if test_end_s is not None:
            valid &= event_times < test_end_s
        y_source = trials["choice"]
        y_label = 1
        window = tuple(getattr(decode_task.config, "DECODE_CHOICE_WINDOW", (-0.1, 0.0)))
    elif task == "feedback":
        event_times = trials["feedback_times"]
        align_label = "feedback"
        if test_end_s is not None:
            valid &= event_times < test_end_s
        y_source = trials["feedback"]
        y_label = 1
        window = tuple(getattr(decode_task.config, "DECODE_FEEDBACK_WINDOW", (0.0, 0.2)))
    else:
        raise ValueError(f"unknown task: {task}")

    idx = np.where(valid)[0]
    if idx.size < 30 or len(cluster_ids) < 3:
        return {
            "skipped": True,
            "reason": "too few trials or clusters",
            "n_trials": int(idx.size),
            "n_clusters": int(len(cluster_ids)),
        }

    x = decode_task.compute_firing_rates(
        spikes["times"], spikes["labels"], cluster_ids, event_times[idx], window=window)
    active = x.std(axis=0) > 1e-6
    if int(np.count_nonzero(active)) < 2:
        return {
            "skipped": True,
            "reason": "too few active clusters",
            "n_trials": int(idx.size),
            "n_clusters": int(len(cluster_ids)),
        }

    y = (y_source[idx] == y_label).astype(np.int32)
    res = decode_task.decode_logistic(
        x[:, active],
        y,
        n_splits=n_splits,
        n_shuffle=n_shuffle,
        quant_bits=quant_bits,
    )
    return {
        "skipped": False,
        "align": align_label,
        "window_s": list(window),
        "balanced_accuracy": float(res["real_acc"]),
        "quant_accs": {str(k): float(v) for k, v in res.get("quant_accs", {}).items()},
        "null_mean": float(res["null_mean"]),
        "null_std": float(res["null_std"]),
        "p_value": float(res["p_value"]),
        "n_trials": int(res["n_trials"]),
        "n_features": int(res["n_features"]),
        "n_nonzero": int(res.get("n_nonzero", 0)),
        "n_clusters": int(len(cluster_ids)),
    }


def _load_wheel_from_one(one, eid: str) -> dict[str, Any]:
    try:
        wheel = one.load_object(eid, "wheel", collection="alf")
        position = np.asarray(wheel["position"], dtype=np.float64)
        timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
    except Exception:
        position = np.asarray(
            one.load_dataset(eid, "_ibl_wheel.position.npy", collection="alf"),
            dtype=np.float64,
        )
        timestamps = np.asarray(
            one.load_dataset(eid, "_ibl_wheel.timestamps.npy", collection="alf"),
            dtype=np.float64,
        )
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    position = position[order]
    keep = np.isfinite(timestamps) & np.isfinite(position)
    return {
        "timestamps": timestamps[keep],
        "position": position[keep],
        "position_file": f"ONE:{eid}:_ibl_wheel.position.npy",
        "timestamps_file": f"ONE:{eid}:_ibl_wheel.timestamps.npy",
    }


def _latest_local_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"no local file matching {pattern} under {root}")
    return matches[-1]


def _load_trials_from_session(session_dir: Path) -> dict[str, Any]:
    import pandas as pd

    table_path = _latest_local_file(session_dir / "alf", "_ibl_trials.table.pqt")
    table = pd.read_parquet(table_path)
    n_trials = len(table)

    def column(name: str, default=np.nan):
        if name in table.columns:
            return table[name].to_numpy()
        return np.full(n_trials, default)

    choice = column("choice")
    feedback = column("feedbackType")
    stim_on = column("stimOn_times")
    feedback_times = column("feedback_times")
    valid = (
        np.isfinite(stim_on)
        & np.isfinite(feedback_times)
        & np.isin(choice, [-1, 1])
        & np.isin(feedback, [-1, 1])
    )
    print(f"  Trials loaded locally: {n_trials} trials", flush=True)
    print(f"  Valid trials: {int(valid.sum())}/{n_trials}", flush=True)
    return {
        "choice": choice,
        "feedback": feedback,
        "stim_on": stim_on,
        "feedback_times": feedback_times,
        "first_movement": column("firstMovement_times"),
        "go_cue": column("goCue_times"),
        "contrast_left": column("contrastLeft"),
        "contrast_right": column("contrastRight"),
        "valid": valid,
        "n_trials": n_trials,
    }


def _load_wheel_from_session(session_dir: Path) -> dict[str, Any]:
    alf_dir = session_dir / "alf"
    position_path = _latest_local_file(alf_dir, "_ibl_wheel.position.npy")
    timestamps_path = _latest_local_file(alf_dir, "_ibl_wheel.timestamps.npy")
    position = np.asarray(np.load(position_path, allow_pickle=False), dtype=np.float64)
    timestamps = np.asarray(np.load(timestamps_path, allow_pickle=False), dtype=np.float64)
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    position = position[order]
    keep = np.isfinite(timestamps) & np.isfinite(position)
    return {
        "timestamps": timestamps[keep],
        "position": position[keep],
        "position_file": str(position_path),
        "timestamps_file": str(timestamps_path),
    }


def _load_reference_from_session(
    session_dir: Path,
    probe: str,
    *,
    sorter: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import pandas as pd

    sorter_dir = session_dir / "alf" / probe / "pykilosort"
    spike_times = np.load(
        _latest_local_file(sorter_dir, "spikes.times.npy"), mmap_mode="r")
    spike_clusters = np.asarray(np.load(
        _latest_local_file(sorter_dir, "spikes.clusters.npy"), mmap_mode="r"),
        dtype=np.int32,
    )
    cluster_channels_saved = np.asarray(np.load(
        _latest_local_file(sorter_dir, "clusters.channels.npy"),
        allow_pickle=False,
    ), dtype=np.int32)
    positions_sorted = np.asarray(np.load(
        _latest_local_file(sorter_dir, "channels.localCoordinates.npy"),
        allow_pickle=False,
    ), dtype=np.float32)
    raw_ind = np.asarray(np.load(
        _latest_local_file(sorter_dir, "channels.rawInd.npy"),
        allow_pickle=False,
    ), dtype=np.int32)
    n_raw = max(int(raw_ind.max()) + 1, 384)
    positions = np.zeros((n_raw, 2), dtype=np.float32)
    positions[raw_ind] = positions_sorted[:len(raw_ind)]

    cluster_ids = np.unique(spike_clusters)
    n_cluster_slots = int(cluster_ids.max()) + 1
    cluster_channels = np.zeros(n_cluster_slots, dtype=np.int32)
    count = min(len(cluster_channels_saved), n_cluster_slots)
    cluster_channels[:count] = cluster_channels_saved[:count]
    valid_channels = cluster_channels < len(raw_ind)
    cluster_channels[valid_channels] = raw_ind[cluster_channels[valid_channels]]

    metrics = pd.read_parquet(_latest_local_file(sorter_dir, "clusters.metrics.pqt"))
    cluster_labels = np.zeros(n_cluster_slots, dtype=np.int32)
    cluster_amplitudes = np.zeros(n_cluster_slots, dtype=np.float32)
    cluster_id_values = (
        metrics["cluster_id"].to_numpy()
        if "cluster_id" in metrics.columns
        else np.arange(len(metrics))
    )
    for row_index, cluster_id_value in enumerate(cluster_id_values):
        cluster_id = int(cluster_id_value)
        if not 0 <= cluster_id < n_cluster_slots:
            continue
        if "label" in metrics.columns:
            value = metrics["label"].iloc[row_index]
            if np.isfinite(value):
                cluster_labels[cluster_id] = 1 if value > 0.8 else (2 if value > 0.5 else 0)
        if "amp_median" in metrics.columns:
            cluster_amplitudes[cluster_id] = float(metrics["amp_median"].iloc[row_index])

    spike_channels = cluster_channels[spike_clusters]
    n_good = int(np.count_nonzero(cluster_labels[cluster_ids] == 1))
    print(
        f"[REF] Loaded local {sorter}: {len(spike_times)} spikes, "
        f"{len(cluster_ids)} clusters ({n_good} good)",
        flush=True,
    )
    ref = {
        "spike_times": spike_times,
        "spike_clusters": spike_clusters,
        "spike_channels": spike_channels,
        "cluster_channels": cluster_channels,
        "cluster_labels": cluster_labels,
        "cluster_amplitudes": cluster_amplitudes,
        "cluster_ids": cluster_ids,
        "n_clusters": len(cluster_ids),
        "n_good": n_good,
        "sorter": sorter,
    }
    return ref, positions, raw_ind


def _load_wheel_helper(script_path: Path):
    spec = importlib.util.spec_from_file_location("wheel_decode_helper", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wheel_metrics(
    *,
    wheel_mod,
    wheel: dict[str, Any],
    trials: dict[str, Any],
    spikes: dict[str, Any],
    cluster_ids: list[int],
    val_end_s: float,
    test_end_s: float | None,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in ("velocity", "speed"):
        out[target] = wheel_mod._run_source_target(
            wheel=wheel,
            trials=trials,
            spikes=spikes,
            cluster_ids=cluster_ids,
            target_mode=target,
            val_end_s=val_end_s,
            test_end_s=test_end_s,
            bin_size_s=wheel_mod.BIN_SIZE_S,
            target_start_s=wheel_mod.TARGET_START_S,
            target_stop_s=wheel_mod.TARGET_STOP_S,
            history_bins=wheel_mod.HISTORY_BINS,
            n_splits=n_splits,
            alpha_grid=wheel_mod.ALPHA_GRID,
            max_iter=1000,
            tol=0.001,
            random_state=random_state,
            nested=True,
        )
    return out



def _build_previous_style_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact table matching the earlier manuscript-style metric convention."""
    ibl = next((s for s in summary.get("sources", [])
                if s.get("name") == "ibl_reference"), {})
    ibl_wheel = ibl.get("wheel", {}) if isinstance(ibl, dict) else {}
    rows: list[dict[str, Any]] = []
    for src in summary.get("sources", []):
        if src.get("name") == "ibl_reference":
            continue
        event = src.get("ks4_event_detection", {})
        decoding = src.get("decoding", {})
        wheel = src.get("wheel", {})
        rows.append({
            "source": src.get("name"),
            "recall_definition": "time+space all pipeline spikes vs KS4 good recall",
            "precision_definition": "time+space assigned pipeline spikes vs KS4 all precision",
            "recall": event.get("good_events_all_pipeline_spikes", {}).get("recall"),
            "precision": event.get("all_events_assigned_pipeline_spikes", {}).get("precision"),
            "feedback_balanced_accuracy": decoding.get("feedback", {}).get("balanced_accuracy"),
            "choice_balanced_accuracy": decoding.get("choice", {}).get("balanced_accuracy"),
            "wheel_velocity_r2_ours": wheel.get("velocity", {}).get("r2"),
            "wheel_speed_r2_ours": wheel.get("speed", {}).get("r2"),
            "wheel_velocity_r2_ibl": ibl_wheel.get("velocity", {}).get("r2"),
            "wheel_speed_r2_ibl": ibl_wheel.get("speed", {}).get("r2"),
        })
    return rows


def _fmt_optional(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct_optional(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)

def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Recording Benchmark Metrics",
        "",
        f"EID: `{summary['eid']}`",
        f"Probe: `{summary['probe']}`",
        f"Calibration: `{summary['calibration_duration_s']} s`",
        f"Validation: `{summary['validation_duration_s']} s`",
        "",
        "## Compact Summary (Previous Style)",
        "",
        "This table preserves the earlier manuscript-style convention: Recall = time+space recall of all pipeline-detected spikes against KS4 good-unit spikes; Precision = time+space precision of assigned pipeline spikes against all KS4 spikes.",
        "",
        "| source | Recall | Precision | Feedback | Choice | Wheel Velocity (Ours, R2) | Wheel Speed (Ours, R2) | Wheel Velocity (IBL, R2) | Wheel Speed (IBL, R2) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("previous_style_summary", _build_previous_style_summary(summary)):
        lines.append(
            f"| {row.get('source', '')} | "
            f"{_fmt_optional(row.get('recall'))} | "
            f"{_fmt_optional(row.get('precision'))} | "
            f"{_fmt_pct_optional(row.get('feedback_balanced_accuracy'))} | "
            f"{_fmt_pct_optional(row.get('choice_balanced_accuracy'))} | "
            f"{_fmt_optional(row.get('wheel_velocity_r2_ours'))} | "
            f"{_fmt_optional(row.get('wheel_speed_r2_ours'))} | "
            f"{_fmt_optional(row.get('wheel_velocity_r2_ibl'))} | "
            f"{_fmt_optional(row.get('wheel_speed_r2_ibl'))} |"
        )
    lines += [
        "",
        "## Choice / Feedback",
        "",
        "| source | task | clusters | balanced acc | 8-bit decoder acc | p | trials |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for src in summary["sources"]:
        for task in ("choice", "feedback"):
            row = src["decoding"].get(task, {})
            if row.get("skipped"):
                lines.append(f"| {src['name']} | {task} | {row.get('n_clusters', '')} | skip | | | {row.get('n_trials', '')} |")
                continue
            q8 = row.get("quant_accs", {}).get("8", "")
            q8_text = f"{q8:.4f}" if isinstance(q8, (float, int)) else ""
            lines.append(
                f"| {src['name']} | {task} | {row['n_clusters']} | "
                f"{row['balanced_accuracy']:.4f} | {q8_text} | "
                f"{row['p_value']:.4g} | {row['n_trials']} |"
            )
    lines += [
        "",
        "## Wheel Movement",
        "",
        "| source | target | clusters | R2 | trials | samples |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for src in summary["sources"]:
        for target, row in src.get("wheel", {}).items():
            if row.get("skipped"):
                lines.append(f"| {src['name']} | {target} | {src['n_clusters_used']} | skip | | |")
                continue
            lines.append(
                f"| {src['name']} | {target} | {src['n_clusters_used']} | "
                f"{row['r2']:.4f} | {row['n_trials']} | {row['n_samples']} |"
            )
    lines += [
        "",
        "## KS4 Full Comparison, Three Layers",
        "",
        "Layer 1 is event detection: time/space coincidence only, no cluster identity.",
        "Layer 2 is good-unit coverage: how many KS4 good units are represented.",
        "Layer 3 is strict sorted-spike fidelity: one-to-one cluster matching with precision/recall.",
        "",
        "### Layer 1: Event Detection",
        "",
        "Layer 1 intentionally separates detection views: time-only vs time+space, all pipeline events vs assigned events, and KS4 all vs KS4 good-unit events.",
        "",
        "| source | match view | pipeline set | KS4 set | precision | recall | matched | pipe events | KS4 events | dt ms | dist um |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    layer1_rows = [
        ("time+space", "all pipeline spikes", "KS4 all", "all_events_all_pipeline_spikes"),
        ("time+space", "assigned pipeline spikes", "KS4 all", "all_events_assigned_pipeline_spikes"),
        ("time+space", "all pipeline spikes", "KS4 good", "good_events_all_pipeline_spikes"),
        ("time+space", "assigned pipeline spikes", "KS4 good", "good_events_assigned_pipeline_spikes"),
        ("time-only", "all pipeline spikes", "KS4 all", "time_only_all_events_all_pipeline_spikes"),
        ("time-only", "assigned pipeline spikes", "KS4 all", "time_only_all_events_assigned_pipeline_spikes"),
    ]
    for src in summary["sources"]:
        if src["name"] == "ibl_reference":
            continue
        event = src.get("ks4_event_detection", {})
        if event.get("skipped"):
            lines.append(f"| {src['name']} | skip | | | | | | | | | |")
            continue
        for view, pipe_set, ref_set, key in layer1_rows:
            row = event.get(key, {})
            dist = row.get("max_dist_um")
            dist_text = "" if dist is None else f"{float(dist):.1f}"
            lines.append(
                f"| {src['name']} | {view} | {pipe_set} | {ref_set} | "
                f"{float(row.get('precision', 0.0)):.4f} | "
                f"{float(row.get('recall', 0.0)):.4f} | "
                f"{int(row.get('matched', 0))} | "
                f"{int(row.get('pipe_events', 0))} | "
                f"{int(row.get('ref_events', 0))} | "
                f"{float(row.get('max_dt_ms', 0.0)):.3f} | {dist_text} |"
            )

    lines += [
        "",
        "### Layer 2: Good-Unit Coverage",
        "",
        "| source | matched good units | coverage any | recall >=0.1 | recall >=0.2 | recall >=0.5 | accuracy >=0.5 | accuracy >=0.8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for src in summary["sources"]:
        if src["name"] == "ibl_reference":
            continue
        ks4 = src.get("ks4_good", {})
        lines.append(
            f"| {src['name']} | {int(ks4.get('n_good_matched', 0))}/{int(ks4.get('n_ref_good', 0))} | "
            f"{float(ks4.get('good_unit_coverage_any', 0.0)):.4f} | "
            f"{float(ks4.get('good_unit_coverage_recall_ge_0p1', 0.0)):.4f} | "
            f"{float(ks4.get('good_unit_coverage_recall_ge_0p2', 0.0)):.4f} | "
            f"{float(ks4.get('good_unit_coverage_recall_ge_0p5', 0.0)):.4f} | "
            f"{float(ks4.get('good_unit_coverage_accuracy_ge_0p5', 0.0)):.4f} | "
            f"{float(ks4.get('good_unit_coverage_accuracy_ge_0p8', 0.0)):.4f} |"
        )

    lines += [
        "",
        "### Layer 3: Strict Sorted-Spike Fidelity",
        "",
        "| source | good global precision | good global recall | mean precision good | mean recall good | global recall | good TP | good pipe total | good ref total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for src in summary["sources"]:
        if src["name"] == "ibl_reference":
            continue
        ks4 = src.get("ks4_good", {})
        mean_recall_good = ks4.get("mean_recall_good", ks4.get("mean_recall", 0.0))
        lines.append(
            f"| {src['name']} | {float(ks4.get('good_global_precision', 0.0)):.4f} | "
            f"{float(ks4.get('good_global_recall', 0.0)):.4f} | "
            f"{float(ks4.get('mean_precision_good', 0.0)):.4f} | "
            f"{float(mean_recall_good):.4f} | "
            f"{float(ks4.get('global_recall', 0.0)):.4f} | "
            f"{int(ks4.get('good_TP', 0))} | {int(ks4.get('good_pipe_total', 0))} | "
            f"{int(ks4.get('good_ref_total', 0))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eid", required=True)
    p.add_argument("--probe", default="probe00")
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--sw8-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--one-cache-dir", type=Path, default=Path("recordings/one_cache"))
    p.add_argument(
        "--local-session-dir",
        type=Path,
        default=None,
        help="Load cached trials, wheel, geometry, and reference sorting directly.",
    )
    p.add_argument("--ks4-cache", type=Path, default=None,
                   help="Optional shared ks4_full_*.npz cache. Defaults to full/sw8 result cache.")
    p.add_argument("--ibl-spike-sorter", default="iblsorter")
    p.add_argument("--cluster-mode", choices=["all", "good"], default="good")
    p.add_argument("--calibration-duration-s", type=float, default=240.0)
    p.add_argument("--validation-duration-s", type=float, default=120.0)
    p.add_argument("--test-end-s", type=float, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-shuffle", type=int, default=200)
    p.add_argument("--quant-bits", default="2,4,6,8")
    p.add_argument("--skip-wheel", action="store_true")
    p.add_argument("--skip-decode", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.one_cache_dir:
        cache_dir = args.one_cache_dir
        if not cache_dir.is_absolute():
            cache_dir = Path(__file__).resolve().parent / cache_dir
        os.environ["BCI_ONE_CACHE_DIR"] = str(cache_dir)

    import config
    import decode_task
    from comparison import load_reference
    from data_loader import connect_one, load_channel_geometry

    config.IBL_SESSION_EID = args.eid
    config.IBL_PROBE_LABEL = args.probe
    config.CALIBRATION_DURATION_S = float(args.calibration_duration_s)
    config.DECODE_N_SPLITS = int(args.n_splits)
    config.DECODE_N_SHUFFLE = int(args.n_shuffle)
    quant_bits = [int(x) for x in args.quant_bits.split(",") if x.strip()]
    config.DECODE_QUANT_BITS = quant_bits

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "recording_benchmark_metrics.partial.json"
    partial_signature = {
        "eid": args.eid,
        "probe": args.probe,
        "calibration_duration_s": float(args.calibration_duration_s),
        "validation_duration_s": float(args.validation_duration_s),
        "cluster_mode": args.cluster_mode,
        "n_splits": int(args.n_splits),
        "n_shuffle": int(args.n_shuffle),
        "quant_bits": quant_bits,
        "skip_wheel": bool(args.skip_wheel),
        "skip_decode": bool(args.skip_decode),
        "local_session_dir": (
            str(args.local_session_dir.absolute())
            if args.local_session_dir is not None else None
        ),
    }
    partial_sources: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            if all(partial.get(key) == value
                   for key, value in partial_signature.items()):
                partial_sources = {
                    str(row["name"]): row
                    for row in partial.get("sources", [])
                    if isinstance(row, dict) and row.get("name")
                }
                print(
                    f"[METRICS] Resuming checkpoint with sources: "
                    f"{sorted(partial_sources)}",
                    flush=True,
                )
            else:
                print("[METRICS] Ignoring checkpoint with different settings", flush=True)
        except Exception as exc:
            print(f"[METRICS] Ignoring unreadable checkpoint: {exc}", flush=True)

    def save_checkpoint(rows: list[dict[str, Any]]) -> None:
        payload = dict(partial_signature)
        payload["sources"] = rows
        _write_json(partial_path, payload)

    one = None
    if args.local_session_dir is not None:
        # Preserve subst aliases; resolve() expands B: back into a path that
        # exceeds the legacy Windows path limit for deeply nested ALF files.
        session_dir = args.local_session_dir.absolute()
        print(f"[METRICS] Local session: {session_dir}", flush=True)
        trials = _load_trials_from_session(session_dir)
    else:
        one = connect_one(silent=True)
        trials = decode_task.load_trials(one, args.eid)
    val_end_s = float(args.calibration_duration_s) + float(args.validation_duration_s)

    full_spikes = decode_task.load_pipeline_spikes(str(args.full_dir))
    sw8_spikes = decode_task.load_pipeline_spikes(str(args.sw8_dir))
    if args.local_session_dir is not None:
        ref, positions, raw_ind = _load_reference_from_session(
            session_dir,
            args.probe,
            sorter=args.ibl_spike_sorter,
        )
    else:
        positions, raw_ind = load_channel_geometry(one, args.eid, args.probe)
        ref = load_reference(one, args.eid, args.probe,
                             spike_sorter=args.ibl_spike_sorter, raw_ind=raw_ind)
    if ref is None:
        raise RuntimeError("IBL reference spikes are unavailable")
    good_mask = np.asarray(ref["cluster_labels"]) == 1
    ref_is_good = np.zeros(int(np.max(ref["spike_clusters"])) + 1, dtype=bool)
    n = min(ref_is_good.size, good_mask.size)
    ref_is_good[:n] = good_mask[:n]
    order = np.argsort(ref["spike_times"], kind="stable")
    ref_spikes = {
        "times": np.asarray(ref["spike_times"], dtype=np.float64)[order],
        "times_samples": np.rint(np.asarray(ref["spike_times"], dtype=np.float64)[order]
                                 * float(config.SAMPLE_RATE)).astype(np.int64),
        "labels": np.asarray(ref["spike_clusters"], dtype=np.int32)[order],
        "channels": np.asarray(ref["spike_channels"], dtype=np.int32)[order],
        "is_good": ref_is_good,
    }

    sources = [
        ("ibl_reference", None, ref_spikes),
        ("sw_full_precision", args.full_dir, full_spikes),
        ("sw_8bit", args.sw8_dir, sw8_spikes),
    ]
    if args.test_end_s is None:
        test_end_s = min(
            float(np.max(full_spikes["times"])),
            float(np.max(sw8_spikes["times"])),
            float(np.max(ref_spikes["times"])),
        )
    else:
        test_end_s = float(args.test_end_s)

    wheel = None
    wheel_mod = None
    if not args.skip_wheel:
        if args.local_session_dir is not None:
            wheel = _load_wheel_from_session(session_dir)
        else:
            wheel = _load_wheel_from_one(one, args.eid)
        wheel_mod = _load_wheel_helper(Path(__file__).resolve().parent / "09_decode_wheel_movement.py")

    ks4_cache = args.ks4_cache or _find_ks4_cache(args.full_dir, args.sw8_dir)

    source_rows = []
    for name, root, spikes in sources:
        cluster_ids = _cluster_ids(spikes, args.cluster_mode)
        row = partial_sources.get(name)
        if row is None:
            row = {
                "name": name,
                "root": root,
                "n_spikes": int(len(spikes["labels"])),
                "n_clusters_total": int(np.unique(
                    spikes["labels"][np.asarray(spikes["labels"]) >= 0]).size),
                "n_clusters_used": int(len(cluster_ids)),
                "decoding": {},
                "wheel": {},
            }
        else:
            row["root"] = root
            row.setdefault("decoding", {})
            row.setdefault("wheel", {})
            print(f"[METRICS] Resuming source: {name}", flush=True)
        source_rows.append(row)
        save_checkpoint(source_rows)

        if not args.skip_decode:
            for task in ("choice", "feedback"):
                if task in row["decoding"]:
                    print(f"[METRICS] Reusing {name} decoding/{task}", flush=True)
                    continue
                print(f"[METRICS] Starting {name} decoding/{task}", flush=True)
                row["decoding"][task] = _decode_task_result(
                    decode_task=decode_task,
                    spikes=spikes,
                    cluster_ids=cluster_ids,
                    trials=trials,
                    task=task,
                    val_end_s=val_end_s,
                    test_end_s=test_end_s,
                    n_splits=args.n_splits,
                    n_shuffle=args.n_shuffle,
                    quant_bits=quant_bits,
                )
                save_checkpoint(source_rows)
                print(f"[METRICS] Finished {name} decoding/{task}", flush=True)

        if wheel is not None and wheel_mod is not None:
            for target in ("velocity", "speed"):
                if target in row["wheel"]:
                    print(f"[METRICS] Reusing {name} wheel/{target}", flush=True)
                    continue
                print(f"[METRICS] Starting {name} wheel/{target}", flush=True)
                row["wheel"][target] = wheel_mod._run_source_target(
                    wheel=wheel,
                    trials=trials,
                    spikes=spikes,
                    cluster_ids=cluster_ids,
                    target_mode=target,
                    val_end_s=val_end_s,
                    test_end_s=test_end_s,
                    bin_size_s=wheel_mod.BIN_SIZE_S,
                    target_start_s=wheel_mod.TARGET_START_S,
                    target_stop_s=wheel_mod.TARGET_STOP_S,
                    history_bins=wheel_mod.HISTORY_BINS,
                    n_splits=args.n_splits,
                    alpha_grid=wheel_mod.ALPHA_GRID,
                    max_iter=1000,
                    tol=0.001,
                    random_state=42,
                    nested=True,
                )
                save_checkpoint(source_rows)
                print(f"[METRICS] Finished {name} wheel/{target}", flush=True)

        if root is not None:
            if "main_decoding_summary" not in row:
                row["main_decoding_summary"] = _load_main_decoding_summary(root)
                save_checkpoint(source_rows)
            if "ks4_good" not in row:
                print(f"[METRICS] Starting {name} KS4 unit metrics", flush=True)
                row["ks4_good"] = _ks4_good_metrics(
                    root,
                    ks4_cache=ks4_cache,
                    positions=positions,
                )
                save_checkpoint(source_rows)
                print(f"[METRICS] Finished {name} KS4 unit metrics", flush=True)
            if (ks4_cache is not None and positions is not None
                    and "ks4_event_detection" not in row):
                print(f"[METRICS] Starting {name} KS4 event metrics", flush=True)
                try:
                    row["ks4_event_detection"] = _ks4_event_detection_metrics(
                        spikes=spikes,
                        ks4_cache=ks4_cache,
                        positions=positions,
                    )
                except Exception as e:
                    row["ks4_event_detection"] = {
                        "skipped": True,
                        "reason": str(e),
                    }
                save_checkpoint(source_rows)
                print(f"[METRICS] Finished {name} KS4 event metrics", flush=True)

    summary = {
        "format": "gui_bci_recording_benchmark.v1",
        "eid": args.eid,
        "probe": args.probe,
        "calibration_duration_s": float(args.calibration_duration_s),
        "validation_duration_s": float(args.validation_duration_s),
        "test_end_s": test_end_s,
        "cluster_mode": args.cluster_mode,
        "n_splits": int(args.n_splits),
        "n_shuffle": int(args.n_shuffle),
        "quant_bits": quant_bits,
        "one_cache_dir": os.environ.get("BCI_ONE_CACHE_DIR", ""),
        "ks4_cache": ks4_cache,
        "sources": source_rows,
    }
    summary["previous_style_summary"] = _build_previous_style_summary(summary)
    _write_json(args.output_dir / "recording_benchmark_metrics.json", summary)
    _write_report(args.output_dir / "recording_benchmark_report.md", summary)
    partial_path.unlink(missing_ok=True)
    print(f"summary -> {args.output_dir / 'recording_benchmark_metrics.json'}")
    print(f"report  -> {args.output_dir / 'recording_benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
