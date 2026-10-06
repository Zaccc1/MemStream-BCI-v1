"""Runtime helpers for crash reporting and user-facing diagnostics."""

from __future__ import annotations

import datetime as _dt
import os
import platform
import sys
import tempfile
import traceback
from typing import Mapping


def get_app_state_dir() -> str:
    """Return a writable app state directory, creating it if needed."""
    candidates = [
        os.path.join(os.path.expanduser("~"), ".bci-pipeline"),
        os.path.join(tempfile.gettempdir(), "bci-pipeline"),
    ]

    last_error = None
    for state_dir in candidates:
        try:
            os.makedirs(state_dir, exist_ok=True)
            probe_path = os.path.join(state_dir, ".write-test")
            with open(probe_path, "w", encoding="utf-8") as f:
                f.write("ok")
            try:
                os.remove(probe_path)
            except OSError:
                pass
            return state_dir
        except OSError as exc:
            last_error = exc

    raise OSError(f"No writable app state directory found: {last_error}")


def format_error_report(title: str,
                        traceback_text: str | None = None,
                        metadata: Mapping[str, object] | None = None) -> str:
    """Create a readable plain-text crash report."""
    if traceback_text is None:
        traceback_text = traceback.format_exc()

    lines = [
        "BCI Neural Processing Pipeline Crash Report",
        "=" * 41,
        f"Title: {title}",
        f"Timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"Working directory: {os.getcwd()}",
    ]

    if metadata:
        for key, value in metadata.items():
            if value not in (None, ""):
                lines.append(f"{key}: {value}")

    lines.extend([
        "",
        "Traceback:",
        traceback_text.rstrip(),
        "",
    ])
    return "\n".join(lines)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_crash_report(report_text: str,
                       output_dir: str | None = None) -> dict[str, object]:
    """Persist a crash report to the global app state dir and optionally output dir.

    Two copies are written for each destination:
    - a stable "latest" path for quick access
    - a timestamped archive path so crash history is preserved
    """
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_errors: list[str] = []

    global_path = None
    global_archive_path = None
    try:
        state_dir = get_app_state_dir()
        global_path = os.path.join(state_dir, "crash.log")
        global_archive_path = os.path.join(state_dir, f"crash_{stamp}.log")
        _write_text(global_path, report_text)
        _write_text(global_archive_path, report_text)
    except OSError as exc:
        write_errors.append(f"Global crash log write failed: {exc}")

    output_path = None
    output_archive_path = None
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "crash_report.txt")
            output_archive_path = os.path.join(output_dir, f"crash_report_{stamp}.txt")
            _write_text(output_path, report_text)
            _write_text(output_archive_path, report_text)
        except OSError as exc:
            output_path = None
            output_archive_path = None
            write_errors.append(f"Output crash report write failed: {exc}")

    return {
        "global_path": global_path,
        "global_archive_path": global_archive_path,
        "output_path": output_path,
        "output_archive_path": output_archive_path,
        "write_errors": write_errors,
    }
