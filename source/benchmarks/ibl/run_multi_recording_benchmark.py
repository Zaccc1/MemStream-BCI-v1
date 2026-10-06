#!/usr/bin/env python3
"""Run the multi-recording GUI-BCI benchmark.

For each EID this creates:

  benchmark_runs/<eid>/
    configs/full_precision.json
    configs/sw_8bit.json
    runs/full_precision/
    runs/sw_8bit/
    metrics/

The pipeline is still the project's original main.py.  This script only
standardizes config snapshots, output locations, and final metric collection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from config_runtime import snapshot_config


PRECISION_KEYS = [
    "PRECISION_ADC",
    "PRECISION_DAC",
    "PRECISION_FIR",
    "PRECISION_WHITEN",
    "PRECISION_MATCH",
    "PRECISION_SA",
    "PRECISION_PCA",
    "PRECISION_ASSIGN",
    "PRECISION_LDA",
]


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "?:??"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = int(round(width * done / total))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _default_one_cache_dir(root: Path) -> Path:
    """Default to this worktree so downloaded recordings stay with the benchmark."""
    return root / "recordings" / "one_cache"


def _subst_mappings() -> dict[str, str]:
    if os.name != "nt":
        return {}
    proc = subprocess.run(
        ["cmd", "/c", "subst"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {}
    mappings: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        drive = left.strip()[:2].upper()
        target = right.strip()
        if len(drive) == 2 and drive.endswith(":") and target:
            mappings[drive] = target
    return mappings


def _subst_query(drive: str) -> str:
    return _subst_mappings().get(drive.upper(), "")


def _subst_create(drive: str, target: Path) -> None:
    subprocess.run(["cmd", "/c", "subst", drive, str(target)],
                   capture_output=True, text=True, check=True)


def _short_cache_path_for_windows(cache_dir: Path, root: Path) -> Path:
    """Use a short subst alias for long OneDrive paths while writing to root."""
    if os.name != "nt":
        return cache_dir
    cache_dir = cache_dir.resolve()
    root = root.resolve()
    # ONE appends lab/subject/date/session/collection and UUID-bearing file
    # names below cache_dir; keep the root much shorter than MAX_PATH.
    if len(str(cache_dir)) < 100:
        return cache_dir
    try:
        rel = cache_dir.relative_to(root)
    except ValueError:
        return cache_dir

    for letter in ("B:", "O:", "P:", "Q:", "R:", "S:"):
        existing = _subst_query(letter)
        if existing:
            if str(root).lower() in existing.lower():
                print(f"[BENCH] Reusing subst {letter} -> {root}")
                return Path(letter + "\\") / rel
            continue
        _subst_create(letter, root)
        print(f"[BENCH] Created subst {letter} -> {root}")
        print(f"[BENCH] ONE cache physical path: {cache_dir}")
        print(f"[BENCH] ONE cache short alias:  {Path(letter + '\\') / rel}")
        return Path(letter + "\\") / rel

    print(f"[BENCH] WARNING: no free subst drive found; using long cache path {cache_dir}")
    return cache_dir


def _short_root_alias_for_windows(short_cache_dir: Path, root: Path) -> Path:
    """Return the subst root alias that points at root, when one is active."""
    if os.name != "nt" or not short_cache_dir.drive:
        return root
    mapped = _subst_query(short_cache_dir.drive)
    if not mapped:
        return root
    try:
        if Path(mapped).resolve() == root.resolve():
            return Path(short_cache_dir.drive + "\\")
    except OSError:
        pass
    return root


def _path_for_subprocess(path: Path, root: Path, alias_root: Path) -> Path:
    """Use short subst paths for child processes while writing under root."""
    if alias_root == root:
        return path
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path
    return alias_root / rel


class ProgressReporter:
    """Small console progress layer around long-running benchmark commands."""

    def __init__(self, total_steps: int, heartbeat_s: float):
        self.total_steps = max(1, int(total_steps))
        self.heartbeat_s = max(5.0, float(heartbeat_s))
        self.completed = 0
        self.run_start = time.time()
        self.stage_start = self.run_start
        self.current_label = ""
        self._durations: list[float] = []

    def _eta_s(self) -> float | None:
        remaining = self.total_steps - self.completed
        if remaining <= 0:
            return 0.0
        if self._durations:
            return float(sum(self._durations) / len(self._durations) * remaining)
        elapsed = time.time() - self.run_start
        if self.completed > 0:
            return float(elapsed / self.completed * remaining)
        return None

    def _line(self, status: str, detail: str = "") -> str:
        pct = 100.0 * self.completed / max(self.total_steps, 1)
        msg = (
            f"[BENCH] {status:<8s} {_bar(self.completed, self.total_steps)} "
            f"{self.completed:>3d}/{self.total_steps:<3d} {pct:5.1f}% | "
            f"elapsed={_format_seconds(time.time() - self.run_start)} | "
            f"ETA={_format_seconds(self._eta_s())} | {self.current_label}"
        )
        if detail:
            msg += f" | {detail}"
        return msg

    def start(self, label: str, command: list[str] | None = None) -> None:
        self.current_label = label
        self.stage_start = time.time()
        print(self._line("START"))
        if command:
            print("[BENCH] command: " + " ".join(str(x) for x in command))

    def heartbeat(self) -> None:
        print(self._line("RUNNING", f"stage_elapsed={_format_seconds(time.time() - self.stage_start)}"))

    def finish(self, detail: str = "") -> None:
        duration = time.time() - self.stage_start
        self.completed += 1
        self._durations.append(duration)
        print(self._line("DONE", f"stage_time={_format_seconds(duration)}" + (f" | {detail}" if detail else "")))

    def fail(self, returncode: int) -> None:
        print(self._line("FAILED", f"returncode={returncode} | stage_elapsed={_format_seconds(time.time() - self.stage_start)}"))


def _safe_eid(eid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in eid)


def _load_eids(args: argparse.Namespace) -> list[str]:
    eids: list[str] = []
    if args.eid:
        eids.extend(args.eid)
    if args.eid_file:
        for line in args.eid_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                eids.append(line)
    seen: set[str] = set()
    unique: list[str] = []
    for eid in eids:
        if eid not in seen:
            seen.add(eid)
            unique.append(eid)
    if not unique:
        raise ValueError("provide at least one --eid or --eid-file")
    return unique


def _write_snapshot(path: Path, values: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=True) + "\n",
                    encoding="utf-8")
    return path


def _seed_sw8_ks4_cache(full_dir: Path, sw8_dir: Path) -> None:
    """Reuse the raw-data-only KS4 reference for the SW 8-bit pipeline."""
    sources = sorted((full_dir / "ks4_cache_full").glob("ks4_full_*s_*ch.npz"))
    if not sources:
        print("[BENCH] No full-precision KS4 cache available to seed SW 8-bit")
        return
    target_dir = sw8_dir / "ks4_cache_full"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = target_dir / source.name
        if target.exists() and target.stat().st_size == source.stat().st_size:
            continue
        target.unlink(missing_ok=True)
        try:
            os.link(source, target)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            method = "copy"
        print(f"[BENCH] Seeded SW 8-bit KS4 cache ({method}): {target}")


def _cleanup_completed_intermediates(eid_dir: Path) -> None:
    """Remove large rebuildable work files after final metrics are durable."""
    metrics_path = eid_dir / "metrics" / "recording_benchmark_metrics.json"
    if not metrics_path.exists():
        print(f"[BENCH] Cleanup skipped; metrics missing: {metrics_path}")
        return

    removed_bytes = 0
    for path in (eid_dir / "runs").rglob("*.bin"):
        if path.name not in {"calibration_ap.bin", "full_recording_ap.bin"}:
            continue
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        removed_bytes += size
        print(f"[BENCH] Removed completed-run temporary: {path}")

    for cache_dir in (eid_dir / "runs").glob("*/ks4_cache_full"):
        for path in cache_dir.iterdir():
            if not path.is_file() or path.stat().st_size < 100 * 1024 * 1024:
                continue
            if path.match("ks4_full_*s_*ch.npz"):
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            removed_bytes += size
            print(f"[BENCH] Removed rebuildable KS4 work array: {path}")

    print(f"[BENCH] Cleanup reclaimed {removed_bytes / (1024 ** 3):.1f} GiB")


def _common_snapshot(
    *,
    base: dict[str, Any],
    eid: str,
    probe: str,
    calibration_duration_s: float,
    one_cache_dir: Path,
    max_test_batches: int,
    run_ks4_full: bool,
) -> dict[str, Any]:
    values = dict(base)
    sample_rate = float(values.get("SAMPLE_RATE", 30_000))
    chunk_ms = float(values.get("CHUNK_DURATION_MS", 100.0))
    chunk_samples = int(sample_rate * chunk_ms / 1000.0)
    values.update(
        {
            "DATA_SOURCE": "ibl",
            "IBL_SESSION_EID": eid,
            "IBL_PROBE_LABEL": probe,
            "IBL_CACHE_DIR": str(one_cache_dir),
            "CALIBRATION_DURATION_S": float(calibration_duration_s),
            "CHUNK_SAMPLES": chunk_samples,
            "N_CALIB_CHUNKS": int(round(float(calibration_duration_s) * sample_rate / chunk_samples)),
            "MAX_TEST_BATCHES": int(max_test_batches),
            "KS4_FULL_ENABLE": bool(run_ks4_full),
            "KS4_CACHE_ENABLE": True,
            "DECODE_N_SPLITS": 5,
        }
    )
    return values


def _full_precision_snapshot(common: dict[str, Any]) -> dict[str, Any]:
    values = dict(common)
    values.update(
        {
            "QUANTIZATION_BACKEND": "fake",
            "QUANT_DIAGNOSTICS": False,
            "QUANT_REFIT_AFTER_RANGE_FREEZE": False,
        }
    )
    for key in PRECISION_KEYS:
        values[key] = 32
    return values


def _sw8_snapshot(common: dict[str, Any]) -> dict[str, Any]:
    values = dict(common)
    values.update(
        {
            "QUANTIZATION_BACKEND": "uint8_int_accum",
            "QUANT_DIAGNOSTICS": True,
            "QUANT_RANGE_PERCENTILE": 99.99,
            "QUANT_RANGE_STAGE_OVERRIDES": {},
            "QUANT_RANGE_SYMMETRIC": False,
            "QUANT_REFIT_AFTER_RANGE_FREEZE": True,
            # KS4 full reference is raw-data dependent, not quantization dependent.
            # Keep the full-reference comparison enabled for sw_8bit too.  When
            # the full-precision run has already populated ks4_cache_full, the
            # pipeline reuses that cache and only writes sw_8bit's own
            # comparison_vs_ks4_full artifact; without this, Layer 2/3 reports
            # have no strict cluster-matching file to read.
            "KS4_FULL_ENABLE": True,
        }
    )
    for key in PRECISION_KEYS:
        values[key] = 8
    return values


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    progress: ProgressReporter,
    label: str,
) -> None:
    progress.start(label, cmd)
    if dry_run:
        progress.finish("dry-run")
        return
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)
    last_heartbeat = time.time()
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        now = time.time()
        if now - last_heartbeat >= progress.heartbeat_s:
            progress.heartbeat()
            last_heartbeat = now
        time.sleep(1.0)
    if rc != 0:
        progress.fail(int(rc))
        raise subprocess.CalledProcessError(int(rc), cmd)
    progress.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eid", action="append", help="IBL EID. Can be repeated.")
    p.add_argument("--eid-file", type=Path, default=None)
    p.add_argument("--probe", default="probe00")
    p.add_argument("--region", default="VISp")
    p.add_argument("--output-root", type=Path, default=Path("benchmark_runs"))
    p.add_argument(
        "--one-cache-dir",
        type=Path,
        default=None,
        help="ONE cache root. Defaults to recordings/one_cache inside this benchmark worktree.",
    )
    p.add_argument("--calibration-duration-s", type=float, default=240.0)
    p.add_argument("--max-test-batches", type=int, default=0,
                   help="0 means full streaming after calibration.")
    p.add_argument("--no-ks4-full", action="store_true",
                   help="Skip KS4 full reference. Metrics will lack KS4-good precision/recall.")
    p.add_argument("--skip-pipeline", action="store_true",
                   help="Only run metric collection from existing output directories.")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--progress-heartbeat-s", type=float, default=120.0,
                   help="Print outer benchmark heartbeat while a subprocess is still running.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_root = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    requested_cache_dir = args.one_cache_dir or _default_one_cache_dir(root)
    physical_one_cache_dir = ((root / requested_cache_dir).resolve()
                              if not requested_cache_dir.is_absolute()
                              else requested_cache_dir)
    one_cache_dir = _short_cache_path_for_windows(physical_one_cache_dir, root)
    subprocess_root = _short_root_alias_for_windows(one_cache_dir, root)
    if subprocess_root != root:
        print(f"[BENCH] Child-process paths will use short root: {subprocess_root}")
    env = os.environ.copy()
    env["BCI_ONE_CACHE_DIR"] = str(one_cache_dir)

    eids = _load_eids(args)
    base = snapshot_config()
    steps_per_eid = 1
    if not args.skip_pipeline:
        steps_per_eid += 2
    if not args.skip_metrics:
        steps_per_eid += 1
    progress = ProgressReporter(
        total_steps=steps_per_eid * len(eids),
        heartbeat_s=args.progress_heartbeat_s,
    )
    for rec_idx, eid in enumerate(eids, start=1):
        eid_dir = output_root / _safe_eid(eid)
        cfg_dir = eid_dir / "configs"
        run_dir = eid_dir / "runs"
        metrics_dir = eid_dir / "metrics"
        full_dir = run_dir / "full_precision"
        sw8_dir = run_dir / "sw_8bit"
        progress.start(f"recording {rec_idx}/{len(eids)} {eid}: prepare configs")
        common = _common_snapshot(
            base=base,
            eid=eid,
            probe=args.probe,
            calibration_duration_s=args.calibration_duration_s,
            one_cache_dir=one_cache_dir,
            max_test_batches=args.max_test_batches,
            run_ks4_full=not args.no_ks4_full,
        )
        full_cfg = _write_snapshot(cfg_dir / "full_precision.json",
                                   _full_precision_snapshot(common))
        sw8_cfg = _write_snapshot(cfg_dir / "sw_8bit.json",
                                  _sw8_snapshot(common))
        full_cfg_arg = _path_for_subprocess(full_cfg, root, subprocess_root)
        sw8_cfg_arg = _path_for_subprocess(sw8_cfg, root, subprocess_root)
        full_dir_arg = _path_for_subprocess(full_dir, root, subprocess_root)
        sw8_dir_arg = _path_for_subprocess(sw8_dir, root, subprocess_root)
        metrics_dir_arg = _path_for_subprocess(metrics_dir, root, subprocess_root)
        progress.finish(f"configs={cfg_dir}")

        if not args.skip_pipeline:
            _run(
                [
                    args.python,
                    "main.py",
                    "--source",
                    "ibl",
                    "--eid",
                    eid,
                    "--probe",
                    args.probe,
                    "--region",
                    args.region,
                    "--config-json",
                    str(full_cfg_arg),
                    "--output",
                    str(full_dir_arg),
                ],
                cwd=root,
                env=env,
                dry_run=args.dry_run,
                progress=progress,
                label=f"recording {rec_idx}/{len(eids)} {eid}: full_precision pipeline",
            )
            if not args.no_ks4_full:
                _seed_sw8_ks4_cache(full_dir, sw8_dir)
            _run(
                [
                    args.python,
                    "main.py",
                    "--source",
                    "ibl",
                    "--eid",
                    eid,
                    "--probe",
                    args.probe,
                    "--region",
                    args.region,
                    "--config-json",
                    str(sw8_cfg_arg),
                    "--output",
                    str(sw8_dir_arg),
                ],
                cwd=root,
                env=env,
                dry_run=args.dry_run,
                progress=progress,
                label=f"recording {rec_idx}/{len(eids)} {eid}: sw_8bit pipeline",
            )

        if not args.skip_metrics:
            _run(
                [
                    args.python,
                    "benchmark_recording_metrics.py",
                    "--eid",
                    eid,
                    "--probe",
                    args.probe,
                    "--full-dir",
                    str(full_dir_arg),
                    "--sw8-dir",
                    str(sw8_dir_arg),
                    "--output-dir",
                    str(metrics_dir_arg),
                    "--one-cache-dir",
                    str(one_cache_dir),
                    "--calibration-duration-s",
                    str(args.calibration_duration_s),
                ],
                cwd=root,
                env=env,
                dry_run=args.dry_run,
                progress=progress,
                label=f"recording {rec_idx}/{len(eids)} {eid}: metrics summary",
            )
            _cleanup_completed_intermediates(eid_dir)
    print(f"[BENCH] ALL DONE elapsed={_format_seconds(time.time() - progress.run_start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
