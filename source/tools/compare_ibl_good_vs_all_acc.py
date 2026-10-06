#!/usr/bin/env python3
"""Compare good-only and all-cluster task decoding on the first ten IBL runs.

This is a lightweight reporting analysis. It reads preserved spike sorting
arrays, reuses the original task windows and five-fold decoder, and does not
rerun spike sorting, wheel decoding, permutation tests, or AUC analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


FIRST_TEN_EIDS = [
    "9b5a1754-ac99-4d53-97d3-35c2f6638507",
    "4fa70097-8101-4f10-b585-db39429c5ed0",
    "9a6e127b-bb07-4be2-92e2-53dd858c2762",
    "8c552ddc-813e-4035-81cc-3971b57efe65",
    "7f150b7c-c261-46e6-9edb-cc391c9d9f03",
    "a4000c2f-fa75-4b3e-8f06-a7cf599b87ad",
    "f304211a-81b1-446f-a435-25e589fe3a5a",
    "ee8b36de-779f-4dea-901f-e0141c95722b",
    "83d85891-bd75-4557-91b4-1cbb5f8bfc9d",
    "1a507308-c63a-4e02-8f32-3239a07dc578",
]

SOURCE_DIRS = {
    "sw_full_precision": "full_precision",
    "sw_8bit": "sw_8bit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=Path("B:/"))
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("B:/benchmark_runs")
    )
    parser.add_argument(
        "--one-cache", type=Path, default=Path("B:/recordings/one_cache")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ibl_first10_good_vs_all_acc_20260809"),
    )
    parser.add_argument(
        "--eids",
        default=",".join(FIRST_TEN_EIDS),
        help="Comma-separated EIDs; defaults to the first completed batch of ten.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _walk_session_record(value: Any, eid: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if str(value.get("id", "")) == eid and value.get("subject"):
            return value
        for child in value.values():
            found = _walk_session_record(child, eid)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _walk_session_record(child, eid)
            if found is not None:
                return found
    return None


def _session_record(one_cache: Path, eid: str) -> dict[str, Any]:
    rest_root = one_cache / ".rest"
    for path in rest_root.iterdir():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if eid not in text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        found = _walk_session_record(payload, eid)
        if found is not None:
            return found
    raise FileNotFoundError(f"no cached session metadata for {eid}")


def _session_dir(one_cache: Path, record: dict[str, Any]) -> Path:
    date = str(record["start_time"])[:10]
    number = f"{int(record.get('number', 1)):03d}"
    return (
        one_cache
        / str(record["lab"])
        / "Subjects"
        / str(record["subject"])
        / date
        / number
    )


def _load_trials(
    *,
    eid: str,
    one_cache: Path,
    metrics_module,
    decode_task,
    one_holder: dict[str, Any],
) -> tuple[dict[str, Any], Path, bool]:
    record = _session_record(one_cache, eid)
    session_dir = _session_dir(one_cache, record)
    try:
        trials = metrics_module._load_trials_from_session(session_dir)
        return trials, session_dir, False
    except FileNotFoundError:
        pass

    if "one" not in one_holder:
        from data_loader import connect_one

        one_holder["one"] = connect_one(silent=True)
    trials = decode_task.load_trials(one_holder["one"], eid)
    return trials, session_dir, True


def _task_samples(
    *,
    trials: dict[str, Any],
    decode_task,
    task: str,
    val_end_s: float,
    test_end_s: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], str]:
    valid = np.asarray(trials["valid"], dtype=bool).copy()
    valid &= np.asarray(trials["stim_on"], dtype=np.float64) >= val_end_s
    valid &= np.isfinite(trials["stim_on"])
    valid &= np.isfinite(trials["feedback_times"])

    if task == "choice":
        event_times, align = decode_task.get_choice_event_times(trials)
        if event_times is None:
            raise RuntimeError("choice event unavailable")
        event_times = np.asarray(event_times, dtype=np.float64)
        valid &= np.isfinite(event_times)
        valid &= event_times < test_end_s
        labels = (np.asarray(trials["choice"])[valid] == 1).astype(np.int32)
        window = tuple(decode_task.config.DECODE_CHOICE_WINDOW)
    elif task == "feedback":
        align = "feedback"
        event_times = np.asarray(trials["feedback_times"], dtype=np.float64)
        valid &= event_times < test_end_s
        labels = (np.asarray(trials["feedback"])[valid] == 1).astype(np.int32)
        window = tuple(decode_task.config.DECODE_FEEDBACK_WINDOW)
    else:
        raise ValueError(task)

    return event_times[valid], labels, window, str(align)


def _count_firing_rates(
    spike_times: np.ndarray,
    spike_labels: np.ndarray,
    cluster_ids: list[int],
    event_times: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    """Count spikes with searchsorted while preserving decode_task semantics."""
    times = np.asarray(spike_times, dtype=np.float64)
    labels = np.asarray(spike_labels, dtype=np.int32)
    if times.size > 1 and np.any(times[1:] < times[:-1]):
        order = np.argsort(times, kind="stable")
        times = times[order]
        labels = labels[order]

    ids = np.asarray(cluster_ids, dtype=np.int32)
    max_id = int(max(int(labels.max(initial=-1)), int(ids.max(initial=-1))))
    id_to_column = np.full(max_id + 1, -1, dtype=np.int32)
    id_to_column[ids] = np.arange(len(ids), dtype=np.int32)
    rates = np.zeros((len(event_times), len(ids)), dtype=np.float32)
    pre, post = window

    starts = np.searchsorted(times, event_times + pre, side="left")
    stops = np.searchsorted(times, event_times + post, side="left")
    for row, (start, stop) in enumerate(zip(starts, stops)):
        trial_labels = labels[start:stop]
        valid = (trial_labels >= 0) & (trial_labels <= max_id)
        columns = id_to_column[trial_labels[valid]]
        columns = columns[columns >= 0]
        if columns.size:
            rates[row] = np.bincount(columns, minlength=len(ids))
    rates /= max(float(post - pre), 1e-6)
    return rates


def _five_fold_acc(
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    active = np.asarray(matrix.std(axis=0) > 1e-6)
    matrix = matrix[:, active]
    scores: list[float] = []
    nonzero: list[int] = []
    for train, test in splits:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(matrix[train])
        test_x = scaler.transform(matrix[test])
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=1000,
            random_state=42,
        )
        model.fit(train_x, labels[train])
        prediction = model.predict(test_x)
        scores.append(float(balanced_accuracy_score(labels[test], prediction)))
        nonzero.append(int(np.count_nonzero(np.abs(model.coef_) > 1e-9)))
    return {
        "acc": float(np.mean(scores)),
        "fold_acc": scores,
        "n_features": int(matrix.shape[1]),
        "n_nonzero_mean": float(np.mean(nonzero)),
    }


def _decode_source(
    *,
    spikes: dict[str, Any],
    trials: dict[str, Any],
    decode_task,
    metrics_module,
    val_end_s: float,
    test_end_s: float,
) -> list[dict[str, Any]]:
    from sklearn.model_selection import StratifiedKFold

    all_ids = metrics_module._cluster_ids(spikes, "all")
    good_ids = metrics_module._cluster_ids(spikes, "good")
    all_column = {cluster_id: index for index, cluster_id in enumerate(all_ids)}
    good_columns = np.asarray([all_column[x] for x in good_ids], dtype=np.int32)
    output: list[dict[str, Any]] = []

    for task in ("choice", "feedback"):
        events, labels, window, align = _task_samples(
            trials=trials,
            decode_task=decode_task,
            task=task,
            val_end_s=val_end_s,
            test_end_s=test_end_s,
        )
        if len(events) < 30 or min(np.bincount(labels)) < 5:
            raise RuntimeError(f"insufficient {task} trials: {len(events)}")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = [(train, test) for train, test in cv.split(events, labels)]
        all_matrix = _count_firing_rates(
            spikes["times"], spikes["labels"], all_ids, events, window
        )
        for mode, matrix, selected_ids in (
            ("good", all_matrix[:, good_columns], good_ids),
            ("all", all_matrix, all_ids),
        ):
            result = _five_fold_acc(matrix, labels, splits)
            result.update(
                {
                    "task": task,
                    "cluster_mode": mode,
                    "n_clusters_selected": int(len(selected_ids)),
                    "n_trials": int(len(events)),
                    "n_positive": int(labels.sum()),
                    "align": align,
                    "window_start_s": float(window[0]),
                    "window_end_s": float(window[1]),
                }
            )
            output.append(result)
    return output


def _write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "good_vs_all_acc.json"
    json_path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    rows = payload["rows"]
    csv_fields = [
        "eid",
        "session",
        "source",
        "task",
        "good_acc",
        "all_acc",
        "delta_all_minus_good",
        "good_clusters_selected",
        "all_clusters_selected",
        "good_active_features",
        "all_active_features",
        "n_trials",
        "n_positive",
        "align",
        "window_start_s",
        "window_end_s",
        "good_fold_acc",
        "all_fold_acc",
    ]
    with (output_dir / "good_vs_all_acc.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["good_fold_acc"] = json.dumps(out["good_fold_acc"])
            out["all_fold_acc"] = json.dumps(out["all_fold_acc"])
            writer.writerow(out)

    lines = [
        "# IBL first ten recordings: Good vs All cluster decoding",
        "",
        "Metric: five-fold mean balanced accuracy (ACC). No AUC was computed.",
        "",
        "| EID | source | task | Good ACC | All ACC | All - Good | Good clusters | All clusters |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['eid']} | {row['source']} | {row['task']} | "
            f"{row['good_acc']:.4f} | {row['all_acc']:.4f} | "
            f"{row['delta_all_minus_good']:+.4f} | "
            f"{row['good_clusters_selected']} | {row['all_clusters_selected']} |"
        )
    lines += ["", "## Means", ""]
    lines += [
        "| source | task | Good mean ACC | All mean ACC | Mean delta | recordings |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["means"]:
        lines.append(
            f"| {row['source']} | {row['task']} | {row['good_acc']:.4f} | "
            f"{row['all_acc']:.4f} | {row['delta_all_minus_good']:+.4f} | "
            f"{row['n_recordings']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    code_root = args.code_root.absolute()
    benchmark_root = args.benchmark_root.absolute()
    one_cache = args.one_cache.absolute()
    sys.path.insert(0, str(code_root))
    os.environ["BCI_ONE_CACHE_DIR"] = str(one_cache)

    import benchmark_recording_metrics as metrics_module
    import config
    import decode_task

    one_holder: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    downloaded_trials: list[str] = []
    eids = [value.strip() for value in args.eids.split(",") if value.strip()]

    for index, eid in enumerate(eids, start=1):
        metric_path = benchmark_root / eid / "metrics" / "recording_benchmark_metrics.json"
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        config.CALIBRATION_DURATION_S = float(metric["calibration_duration_s"])
        config.DECODE_N_SPLITS = 5
        trials, session_dir, downloaded = _load_trials(
            eid=eid,
            one_cache=one_cache,
            metrics_module=metrics_module,
            decode_task=decode_task,
            one_holder=one_holder,
        )
        if downloaded:
            downloaded_trials.append(eid)
        val_end_s = float(metric["calibration_duration_s"]) + float(
            metric["validation_duration_s"]
        )
        test_end_s = float(metric["test_end_s"])
        session = str(session_dir.relative_to(one_cache))
        print(f"[{index}/{len(eids)}] {eid} | {session}", flush=True)

        source_by_name = {source["name"]: source for source in metric["sources"]}
        for source_name, run_name in SOURCE_DIRS.items():
            run_dir = benchmark_root / eid / "runs" / run_name
            spikes = decode_task.load_pipeline_spikes(str(run_dir))
            results = _decode_source(
                spikes=spikes,
                trials=trials,
                decode_task=decode_task,
                metrics_module=metrics_module,
                val_end_s=val_end_s,
                test_end_s=test_end_s,
            )
            good_saved = source_by_name[source_name]["decoding"]
            for task in ("choice", "feedback"):
                good = next(
                    row
                    for row in results
                    if row["task"] == task and row["cluster_mode"] == "good"
                )
                all_result = next(
                    row
                    for row in results
                    if row["task"] == task and row["cluster_mode"] == "all"
                )
                saved_acc = float(good_saved[task]["balanced_accuracy"])
                if abs(saved_acc - good["acc"]) > 1e-10:
                    raise RuntimeError(
                        f"good ACC mismatch for {eid}/{source_name}/{task}: "
                        f"saved={saved_acc}, rerun={good['acc']}"
                    )
                raw_rows.append(
                    {
                        "eid": eid,
                        "session": session,
                        "source": source_name,
                        "task": task,
                        "good_acc": good["acc"],
                        "all_acc": all_result["acc"],
                        "delta_all_minus_good": all_result["acc"] - good["acc"],
                        "good_clusters_selected": good["n_clusters_selected"],
                        "all_clusters_selected": all_result["n_clusters_selected"],
                        "good_active_features": good["n_features"],
                        "all_active_features": all_result["n_features"],
                        "n_trials": good["n_trials"],
                        "n_positive": good["n_positive"],
                        "align": good["align"],
                        "window_start_s": good["window_start_s"],
                        "window_end_s": good["window_end_s"],
                        "good_fold_acc": good["fold_acc"],
                        "all_fold_acc": all_result["fold_acc"],
                    }
                )
            print(f"  finished {source_name}", flush=True)

    means = []
    for source in SOURCE_DIRS:
        for task in ("choice", "feedback"):
            subset = [
                row
                for row in raw_rows
                if row["source"] == source and row["task"] == task
            ]
            means.append(
                {
                    "source": source,
                    "task": task,
                    "good_acc": float(np.mean([row["good_acc"] for row in subset])),
                    "all_acc": float(np.mean([row["all_acc"] for row in subset])),
                    "delta_all_minus_good": float(
                        np.mean([row["delta_all_minus_good"] for row in subset])
                    ),
                    "n_recordings": len(subset),
                }
            )

    payload = {
        "format": "gui_bci.good_vs_all_acc.v1",
        "metric": "five_fold_mean_balanced_accuracy",
        "auc_computed": False,
        "sources": list(SOURCE_DIRS),
        "tasks": ["choice", "feedback"],
        "eids": eids,
        "downloaded_trial_eids": downloaded_trials,
        "rows": raw_rows,
        "means": means,
    }
    _write_outputs(args.output_dir, payload)
    print(f"Done: {args.output_dir.absolute()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
