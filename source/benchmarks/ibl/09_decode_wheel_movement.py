#!/usr/bin/env python3
"""Decode IBL wheel movement traces from IBL/full/sw8/hardware spike events.

Paper-style target construction:
  - align trials to first wheel movement
  - target bins: non-overlapping 20 ms bins from -200 ms to +1000 ms
  - neural history: W=10 causal bins ending at the target bin
  - model: Lasso regression, fit_intercept=True, tol=0.001, max_iter=1000

This script is intentionally source-comparison oriented.  It reports R2 for
wheel velocity and speed for each source/cluster set, but does not implement
the paper's imposter-session null distribution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CODE_DIR = Path("/home/wyzhao/git/hardware_export/analysis/Code_for_analysis")
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
    "/home/wyzhao/git/6.20_TEST1/data/"
    "clean_recording_tail3_sharded_0.000s_4266545ms"
)
DEFAULT_HW_ROOT = Path(
    "/home/wyzhao/git/hardware_export/analysis/vmm/"
    "(U)full_recording_hw12_whitegate4_topkfix_cal5s_streamrest_20260626/"
    "actual_events"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/wyzhao/git/hardware_export/analysis/Code_for_analysis/"
    "results_wheel_movement_decoding_newlong_20260704"
)

BIN_SIZE_S = 0.020
TARGET_START_S = -0.200
TARGET_STOP_S = 1.000
HISTORY_BINS = 10
ALPHA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


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


def _load_source_helpers(code_dir: Path):
    helper_path = code_dir / "05_compare_decoding_sources.py"
    spec = importlib.util.spec_from_file_location("decode_source_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load helper module from {helper_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _download_wheel(eid: str, *, base_url: str = "https://openalyx.internationalbrainlab.org") -> list[Path]:
    from one.api import ONE

    one = ONE(base_url=base_url, silent=True)
    datasets = [
        "_ibl_wheel.position.npy",
        "_ibl_wheel.timestamps.npy",
        "_ibl_wheelMoves.intervals.npy",
        "_ibl_wheelMoves.peakAmplitude.npy",
    ]
    loaded = one.load_datasets(eid, datasets, collections="alf", download_only=True)
    return [Path(p) for p in loaded[0] if p is not None]


def _find_alf_file(session_root: Path, filename: str) -> Path:
    candidates = sorted((session_root / "alf").glob(f"**/{filename}"))
    if candidates:
        # Prefer the top-level alf object when both top-level and revisioned
        # files are present.
        for p in candidates:
            if p.parent == session_root / "alf":
                return p
        return candidates[-1]
    raise FileNotFoundError(f"{filename} not found under {session_root / 'alf'}")


def _load_wheel(session_root: Path) -> dict[str, Any]:
    pos_path = _find_alf_file(session_root, "_ibl_wheel.position.npy")
    ts_path = _find_alf_file(session_root, "_ibl_wheel.timestamps.npy")
    move_path = _find_alf_file(session_root, "_ibl_wheelMoves.intervals.npy")
    amp_path = _find_alf_file(session_root, "_ibl_wheelMoves.peakAmplitude.npy")
    position = np.asarray(np.load(pos_path), dtype=np.float64)
    timestamps = np.asarray(np.load(ts_path), dtype=np.float64)
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    position = position[order]
    keep = np.isfinite(timestamps) & np.isfinite(position)
    timestamps = timestamps[keep]
    position = position[keep]
    if timestamps.size < 2:
        raise ValueError("wheel trace has fewer than two valid samples")
    return {
        "timestamps": timestamps,
        "position": position,
        "moves_intervals": np.asarray(np.load(move_path), dtype=np.float64),
        "moves_peak_amplitude": np.asarray(np.load(amp_path), dtype=np.float64),
        "position_file": pos_path,
        "timestamps_file": ts_path,
        "moves_intervals_file": move_path,
        "moves_peak_amplitude_file": amp_path,
    }


def _wheel_targets(
    wheel: dict[str, Any],
    event_times: np.ndarray,
    *,
    target_mode: str,
    bin_size_s: float,
    target_start_s: float,
    target_stop_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(wheel["timestamps"], dtype=np.float64)
    position = np.asarray(wheel["position"], dtype=np.float64)
    n_bins = int(round((target_stop_s - target_start_s) / bin_size_s))
    rel_edges = target_start_s + np.arange(n_bins + 1, dtype=np.float64) * bin_size_s
    edge_times = event_times[:, None] + rel_edges[None, :]
    flat_edges = edge_times.ravel()
    edge_pos = np.interp(flat_edges, timestamps, position, left=np.nan, right=np.nan)
    edge_pos = edge_pos.reshape(event_times.size, n_bins + 1)
    velocity = np.diff(edge_pos, axis=1) / float(bin_size_s)
    if target_mode == "speed":
        target = np.abs(velocity)
    elif target_mode == "velocity":
        target = velocity
    else:
        raise ValueError(f"unsupported target_mode: {target_mode}")
    return target.astype(np.float32), rel_edges[:-1] + 0.5 * bin_size_s


def _select_trials(
    trials: dict[str, Any],
    wheel: dict[str, Any],
    *,
    val_end_s: float,
    test_end_s: float | None,
    target_start_s: float,
    target_stop_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    first_movement = np.asarray(trials["first_movement"], dtype=np.float64)
    stim_on = np.asarray(trials["stim_on"], dtype=np.float64)
    feedback_times = np.asarray(trials["feedback_times"], dtype=np.float64)
    valid = np.asarray(trials["valid"], dtype=bool).copy()
    valid &= np.isfinite(first_movement)
    valid &= np.isfinite(stim_on)
    valid &= np.isfinite(feedback_times)
    valid &= stim_on >= val_end_s
    if test_end_s is not None:
        valid &= stim_on < test_end_s
        valid &= first_movement < test_end_s
    ts = np.asarray(wheel["timestamps"], dtype=np.float64)
    valid &= first_movement + target_start_s >= ts[0]
    valid &= first_movement + target_stop_s <= ts[-1]
    idx = np.where(valid)[0]
    return idx, first_movement[idx]


def _build_sparse_design(
    *,
    spike_times: np.ndarray,
    spike_labels: np.ndarray,
    cluster_ids: list[int],
    event_times: np.ndarray,
    bin_size_s: float,
    target_start_s: float,
    target_stop_s: float,
    history_bins: int,
):
    from scipy import sparse

    spike_times = np.asarray(spike_times, dtype=np.float64)
    spike_labels = np.asarray(spike_labels, dtype=np.int32)
    cluster_ids_arr = np.asarray(cluster_ids, dtype=np.int32)
    n_trials = int(event_times.size)
    n_clusters = int(cluster_ids_arr.size)
    n_target_bins = int(round((target_stop_s - target_start_s) / bin_size_s))
    n_feature_bins = n_target_bins + history_bins
    n_samples = n_trials * n_target_bins
    n_features = (history_bins + 1) * n_clusters

    max_label = int(max(int(spike_labels.max(initial=0)), int(cluster_ids_arr.max(initial=0))))
    label_to_col = np.full(max_label + 1, -1, dtype=np.int32)
    label_to_col[cluster_ids_arr] = np.arange(n_clusters, dtype=np.int32)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    feature_start = target_start_s - history_bins * bin_size_s
    feature_duration = n_feature_bins * bin_size_s

    for trial_i, event_t in enumerate(event_times):
        lo_t = event_t + feature_start
        hi_t = lo_t + feature_duration
        lo = np.searchsorted(spike_times, lo_t, side="left")
        hi = np.searchsorted(spike_times, hi_t, side="left")
        if hi <= lo:
            continue
        labels = spike_labels[lo:hi]
        times = spike_times[lo:hi]
        label_ok = (labels >= 0) & (labels <= max_label)
        if not np.any(label_ok):
            continue
        labels = labels[label_ok]
        times = times[label_ok]
        cluster_cols = label_to_col[labels]
        ok = cluster_cols >= 0
        if not np.any(ok):
            continue
        cluster_cols = cluster_cols[ok]
        rel = times[ok] - lo_t
        bin_idx = np.floor(rel / bin_size_s).astype(np.int32)
        ok_bin = (bin_idx >= 0) & (bin_idx < n_feature_bins)
        if not np.any(ok_bin):
            continue
        bin_idx = bin_idx[ok_bin]
        cluster_cols = cluster_cols[ok_bin]

        counts = np.zeros((n_feature_bins, n_clusters), dtype=np.float32)
        np.add.at(counts, (bin_idx, cluster_cols), 1.0)
        for target_i in range(n_target_bins):
            sample_row = trial_i * n_target_bins + target_i
            window = counts[target_i : target_i + history_bins + 1]
            nz_bin, nz_col = np.nonzero(window)
            if nz_bin.size == 0:
                continue
            rows.append(np.full(nz_bin.size, sample_row, dtype=np.int32))
            cols.append((nz_bin * n_clusters + nz_col).astype(np.int32))
            data.append(window[nz_bin, nz_col].astype(np.float32))

    if rows:
        x = sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_samples, n_features),
            dtype=np.float32,
        )
    else:
        x = sparse.csr_matrix((n_samples, n_features), dtype=np.float32)
    trial_ids = np.repeat(np.arange(n_trials, dtype=np.int32), n_target_bins)
    bin_ids = np.tile(np.arange(n_target_bins, dtype=np.int32), n_trials)
    return x, trial_ids, bin_ids


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if den <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / den)


def _fit_predict_lasso(
    x_train,
    y_train: np.ndarray,
    x_test,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(with_mean=False),
        Lasso(alpha=float(alpha), fit_intercept=True, max_iter=max_iter, tol=tol),
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=np.float64)


def _decode_lasso_nested(
    x,
    y: np.ndarray,
    trial_ids: np.ndarray,
    bin_ids: np.ndarray,
    *,
    n_trials: int,
    n_splits: int,
    alpha_grid: list[float],
    max_iter: int,
    tol: float,
    random_state: int,
    nested: bool,
) -> dict[str, Any]:
    from sklearn.model_selection import KFold

    y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_flat)
    if not np.any(finite):
        return {"skipped": True, "reason": "no finite wheel targets"}
    x = x[finite]
    y_flat = y_flat[finite]
    trial_ids = trial_ids[finite]
    bin_ids = bin_ids[finite]

    unique_trials = np.arange(n_trials, dtype=np.int32)
    folds = min(int(n_splits), int(unique_trials.size))
    if folds < 2:
        return {"skipped": True, "reason": "too few trials"}
    outer = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    pred = np.full(y_flat.shape, np.nan, dtype=np.float64)
    chosen_alphas: list[float] = []

    for outer_train_trial_idx, outer_test_trial_idx in outer.split(unique_trials):
        train_trials = unique_trials[outer_train_trial_idx]
        test_trials = unique_trials[outer_test_trial_idx]
        train_mask = np.isin(trial_ids, train_trials)
        test_mask = np.isin(trial_ids, test_trials)

        if nested and len(alpha_grid) > 1 and train_trials.size >= folds:
            inner = KFold(n_splits=folds, shuffle=True, random_state=random_state + 17)
            alpha_scores: list[tuple[float, float]] = []
            for alpha in alpha_grid:
                scores: list[float] = []
                for inner_train_idx, inner_val_idx in inner.split(train_trials):
                    inner_train_trials = train_trials[inner_train_idx]
                    inner_val_trials = train_trials[inner_val_idx]
                    inner_train_mask = np.isin(trial_ids, inner_train_trials)
                    inner_val_mask = np.isin(trial_ids, inner_val_trials)
                    y_hat = _fit_predict_lasso(
                        x[inner_train_mask],
                        y_flat[inner_train_mask],
                        x[inner_val_mask],
                        alpha=alpha,
                        max_iter=max_iter,
                        tol=tol,
                    )
                    scores.append(_r2(y_flat[inner_val_mask], y_hat))
                alpha_scores.append((float(np.mean(scores)), float(alpha)))
            best_alpha = max(alpha_scores, key=lambda item: item[0])[1]
        else:
            best_alpha = float(alpha_grid[0])

        chosen_alphas.append(float(best_alpha))
        pred[test_mask] = _fit_predict_lasso(
            x[train_mask],
            y_flat[train_mask],
            x[test_mask],
            alpha=best_alpha,
            max_iter=max_iter,
            tol=tol,
        )

    ok = np.isfinite(pred)
    per_bin: dict[str, float] = {}
    for b in np.unique(bin_ids[ok]):
        m = ok & (bin_ids == b)
        if np.count_nonzero(m) >= 3:
            per_bin[str(int(b))] = _r2(y_flat[m], pred[m])
    return {
        "skipped": False,
        "r2": _r2(y_flat[ok], pred[ok]),
        "n_samples": int(np.count_nonzero(ok)),
        "n_trials": int(unique_trials.size),
        "n_features": int(x.shape[1]),
        "target_mean": float(np.mean(y_flat[ok])),
        "target_std": float(np.std(y_flat[ok])),
        "chosen_alphas": chosen_alphas,
        "chosen_alpha_median": float(np.median(chosen_alphas)),
        "per_bin_r2": per_bin,
    }


def _run_source_target(
    *,
    wheel: dict[str, Any],
    trials: dict[str, Any],
    spikes: dict[str, Any],
    cluster_ids: list[int],
    target_mode: str,
    val_end_s: float,
    test_end_s: float | None,
    bin_size_s: float,
    target_start_s: float,
    target_stop_s: float,
    history_bins: int,
    n_splits: int,
    alpha_grid: list[float],
    max_iter: int,
    tol: float,
    random_state: int,
    nested: bool,
) -> dict[str, Any]:
    idx, event_times = _select_trials(
        trials,
        wheel,
        val_end_s=val_end_s,
        test_end_s=test_end_s,
        target_start_s=target_start_s,
        target_stop_s=target_stop_s,
    )
    if idx.size < max(2, n_splits):
        return {"skipped": True, "reason": "too few wheel trials", "n_trials": int(idx.size)}
    targets, rel_bin_centers = _wheel_targets(
        wheel,
        event_times,
        target_mode=target_mode,
        bin_size_s=bin_size_s,
        target_start_s=target_start_s,
        target_stop_s=target_stop_s,
    )
    x, trial_ids, bin_ids = _build_sparse_design(
        spike_times=spikes["times"],
        spike_labels=spikes["labels"],
        cluster_ids=cluster_ids,
        event_times=event_times,
        bin_size_s=bin_size_s,
        target_start_s=target_start_s,
        target_stop_s=target_stop_s,
        history_bins=history_bins,
    )
    res = _decode_lasso_nested(
        x,
        targets.reshape(-1),
        trial_ids,
        bin_ids,
        n_trials=int(idx.size),
        n_splits=n_splits,
        alpha_grid=alpha_grid,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        nested=nested,
    )
    res.update(
        {
            "target_mode": target_mode,
            "trial_indices": idx,
            "rel_bin_centers_s": rel_bin_centers,
            "bin_size_s": float(bin_size_s),
            "target_window_s": [float(target_start_s), float(target_stop_s)],
            "history_bins": int(history_bins),
        }
    )
    return res


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Wheel Movement Decoding",
        "",
        f"trial_source: `{summary['trial_source']}`",
        f"wheel_position: `{summary['wheel_position_file']}`",
        f"window_s: `{summary['target_window_s']}`",
        f"bin_size_s: `{summary['bin_size_s']}`",
        f"history_bins: `{summary['history_bins']}`",
        f"calibration_duration_s: `{summary['calibration_duration_s']}`",
        f"validation_duration_s: `{summary['validation_duration_s']}`",
        f"test_end_s: `{summary['test_end_s']}`",
        f"model: `{summary['model']}`",
        f"nested_alpha_selection: `{summary['nested_alpha_selection']}`",
        "",
        "| source | cluster set | target | spikes | clusters | R2 | trials | samples | features | alpha median |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for src in summary["sources"]:
        for set_name, set_row in src["cluster_sets"].items():
            for target, row in set_row["targets"].items():
                if row.get("skipped"):
                    lines.append(
                        f"| {src['name']} | {set_name} | {target} | {src['n_spikes']} | "
                        f"{set_row['n_clusters']} | skip | | | | |"
                    )
                    continue
                lines.append(
                    f"| {src['name']} | {set_name} | {target} | {src['n_spikes']} | "
                    f"{set_row['n_clusters']} | {row['r2']:.4f} | {row['n_trials']} | "
                    f"{row['n_samples']} | {row['n_features']} | {row['chosen_alpha_median']:.5g} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR)
    p.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    p.add_argument("--ibl-session-root", type=Path, default=DEFAULT_IBL_SESSION_ROOT)
    p.add_argument("--eid", default=None, help="EID to download wheel files for when --download-wheel is set.")
    p.add_argument("--download-wheel", action="store_true")
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
    p.add_argument("--cluster-set", choices=["all", "good", "both"], default="good")
    p.add_argument("--target", action="append", choices=["velocity", "speed"], default=None)
    p.add_argument("--sample-rate", type=float, default=30000.0)
    p.add_argument("--calibration-duration-s", type=float, default=240.0)
    p.add_argument("--validation-duration-s", type=float, default=120.0)
    p.add_argument("--test-end-s", type=float, default=4266.54)
    p.add_argument("--bin-size-s", type=float, default=BIN_SIZE_S)
    p.add_argument("--target-start-s", type=float, default=TARGET_START_S)
    p.add_argument("--target-stop-s", type=float, default=TARGET_STOP_S)
    p.add_argument("--history-bins", type=int, default=HISTORY_BINS)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--alpha-grid", default=",".join(str(x) for x in ALPHA_GRID))
    p.add_argument("--fixed-alpha", action="store_true", help="Use the first alpha only; skip nested alpha selection.")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=0.001)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="Objective did not converge.*")
    warnings.filterwarnings("ignore", message="With alpha=0.*")

    if args.download_wheel:
        if not args.eid:
            raise ValueError("--eid is required with --download-wheel")
        paths = _download_wheel(args.eid)
        for p in paths:
            print(f"downloaded {p}")

    helpers = _load_source_helpers(args.code_dir)
    sys.path.insert(0, str(args.clean_root))
    from merging import compute_cluster_quality  # noqa: PLC0415

    selected = args.source or ["ibl_reference", "sw_full_precision", "sw_8bits", "hardware"]
    targets = args.target or ["velocity", "speed"]
    alpha_grid = [float(x) for x in re.split(r"[, ]+", args.alpha_grid.strip()) if x]
    if not alpha_grid:
        raise ValueError("--alpha-grid cannot be empty")
    val_end_s = float(args.calibration_duration_s) + float(args.validation_duration_s)
    trials = helpers._load_trials_local(args.ibl_session_root)
    wheel = _load_wheel(args.ibl_session_root)
    source_specs = {
        "ibl_reference": ("ibl_reference", args.ibl_session_root),
        "sw_full_precision": ("sw_full_precision", args.full_root),
        "sw_8bits": ("sw_8bits", args.sw8_root),
        "hardware": ("hardware", args.hw_root),
    }

    sources: list[dict[str, Any]] = []
    for name in selected:
        if name == "ibl_reference":
            spikes = helpers._load_ibl_reference(args.ibl_session_root, sample_rate=args.sample_rate)
        else:
            root = source_specs[name][1]
            spikes = helpers._load_sharded_events(
                root,
                sample_rate=args.sample_rate,
                time_offset_s=helpers._parse_start_s(root),
            )
        sets_all = helpers._cluster_sets(spikes, compute_cluster_quality, args.sample_rate)
        if args.cluster_set == "both":
            set_items = sets_all.items()
        elif args.cluster_set == "all":
            set_items = [("all", sets_all["all"])]
        else:
            set_items = [("good", sets_all["good"])]
        cluster_sets: dict[str, Any] = {}
        for set_name, cluster_ids in set_items:
            target_rows: dict[str, Any] = {}
            for target_mode in targets:
                print(f"[RUN] {name} {set_name} {target_mode}: clusters={len(cluster_ids)}")
                target_rows[target_mode] = _run_source_target(
                    wheel=wheel,
                    trials=trials,
                    spikes=spikes,
                    cluster_ids=cluster_ids,
                    target_mode=target_mode,
                    val_end_s=val_end_s,
                    test_end_s=args.test_end_s,
                    bin_size_s=args.bin_size_s,
                    target_start_s=args.target_start_s,
                    target_stop_s=args.target_stop_s,
                    history_bins=args.history_bins,
                    n_splits=args.n_splits,
                    alpha_grid=alpha_grid,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    random_state=args.random_state,
                    nested=not args.fixed_alpha,
                )
            cluster_sets[set_name] = {
                "n_clusters": int(len(cluster_ids)),
                "targets": target_rows,
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
        "format": "wheel_movement_decoding_sources.v1",
        "clean_root": args.clean_root,
        "trial_source": trials["source_file"],
        "wheel_position_file": wheel["position_file"],
        "wheel_timestamps_file": wheel["timestamps_file"],
        "sample_rate": float(args.sample_rate),
        "bin_size_s": float(args.bin_size_s),
        "target_window_s": [float(args.target_start_s), float(args.target_stop_s)],
        "history_bins": int(args.history_bins),
        "calibration_duration_s": float(args.calibration_duration_s),
        "validation_duration_s": float(args.validation_duration_s),
        "test_end_s": float(args.test_end_s) if args.test_end_s is not None else None,
        "n_splits": int(args.n_splits),
        "alpha_grid": alpha_grid,
        "nested_alpha_selection": not args.fixed_alpha,
        "model": "StandardScaler(with_mean=False) + sklearn Lasso(fit_intercept=True, tol=0.001, max_iter=1000)",
        "sources": sources,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "wheel_movement_decoding_summary.json", summary)
    _write_report(args.output_dir / "wheel_movement_decoding_report.md", summary)
    print(f"summary -> {args.output_dir / 'wheel_movement_decoding_summary.json'}")
    print(f"report  -> {args.output_dir / 'wheel_movement_decoding_report.md'}")
    for src in sources:
        for set_name, set_row in src["cluster_sets"].items():
            for target_mode, row in set_row["targets"].items():
                if row.get("skipped"):
                    print(f"{src['name']} {set_name} {target_mode}: skipped")
                else:
                    print(
                        f"{src['name']} {set_name} {target_mode}: "
                        f"R2={row['r2']:.4f} trials={row['n_trials']} "
                        f"samples={row['n_samples']} features={row['n_features']}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
