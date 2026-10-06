"""Pipeline execution with subprocess-backed run control panel."""
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from gui.qt_compat import (
    QApplication,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar,
    QPlainTextEdit, QLineEdit, QLabel, QFileDialog, QCheckBox,
    QFormLayout, QProcess, pyqtSignal, QFont, QTextCursor, QMessageBox,
)

from config_runtime import save_config_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = str(PROJECT_ROOT / "main.py")
ERROR_REPORT_MARKER = "BCI Neural Processing Pipeline Crash Report"
BACKGROUND_PROCESSES = []


# ---------- Progress parser ----------

STAGE_PATTERNS = [
    (re.compile(r"\[IBL\] Connect|\[LOCAL\] Open", re.IGNORECASE), 5),
    (re.compile(r"CALIBRATION", re.IGNORECASE), 10),
    (re.compile(r"CALI-CACHE", re.IGNORECASE), 18),
    (re.compile(r"[Tt]emplate"), 20),
    (re.compile(r"CLUSTERING|SpikeClustering", re.IGNORECASE), 30),
    (re.compile(r"HW-EXPORT|HARDWARE GOLDEN BATCH EXPORT", re.IGNORECASE), 85),
    (re.compile(r"Batch\s+(\d+)\s*/\s*(\d+)"), None),  # dynamic
    (re.compile(r"CLUSTER MERGING|merge_clusters", re.IGNORECASE), 80),
    (re.compile(r"[Rr]eference comparison|compare_with_reference"), 85),
    (re.compile(r"VALIDATION|DECODING TASK|run_decode_task", re.IGNORECASE), 90),
    (re.compile(r"\[MAIN\]\s*Done|Pipeline complete", re.IGNORECASE), 100),
]


def parse_progress(text: str, current: int) -> int:
    """Parse a log line and return updated progress percentage."""
    for pattern, value in STAGE_PATTERNS:
        m = pattern.search(text)
        if m:
            if value is not None:
                return max(current, value)
            # Batch X/Y pattern — compute dynamic progress
            try:
                batch = int(m.group(1))
                total = int(m.group(2))
                if total > 0:
                    pct = 30 + int((batch / total) * 50)
                    return max(current, min(pct, 80))
            except (IndexError, ValueError):
                pass
    return current


# ---------- Run panel ----------

class RunPanel(QWidget):
    pipeline_finished = pyqtSignal(str)   # output_dir
    status_message = pyqtSignal(str)

    def __init__(self, config_panel, parent=None):
        super().__init__(parent)
        self.config_panel = config_panel
        self._process = None
        self._process_output_dir = None
        self._config_snapshot_path = None
        self._stop_requested = False
        self._captured_output = ""
        self._progress_value = 0
        self._start_time = None
        self._last_error_report = ""

        layout = QVBoxLayout(self)

        # --- Session info (read from config) ---
        import config as cfg
        self.session_label = QLabel(
            f"Session: {cfg.IBL_SESSION_EID or '<not set>'}  |  Probe: {cfg.IBL_PROBE_LABEL}"
        )
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)

        # --- Output directory ---
        form = QFormLayout()
        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit("results_ibl")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_dir)
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(btn_browse)
        form.addRow("Output Directory:", dir_row)
        layout.addLayout(form)

        self.chk_calibration_only = QCheckBox(
            "Calibration only (save quantization range plots, skip streaming)"
        )
        layout.addWidget(self.chk_calibration_only)

        self.chk_hardware_export = QCheckBox(
            "Hardware export (calibrate, export one batch, skip streaming)"
        )
        layout.addWidget(self.chk_hardware_export)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Pipeline")
        self.btn_stop = QPushButton("Stop")
        self.btn_copy_report = QPushButton("Copy Crash Report")
        self.btn_stop.setEnabled(False)
        self.btn_copy_report.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_copy_report.clicked.connect(self._copy_crash_report)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_copy_report)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.chk_close_kills_process = QCheckBox(
            "Close window ends background Python process"
        )
        self.chk_close_kills_process.setChecked(True)
        layout.addWidget(self.chk_close_kills_process)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.stage_label = QLabel("Idle")
        layout.addWidget(self.stage_label)

        # --- Log console ---
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Menlo", 11))
        self.log_console.setMaximumBlockCount(50000)
        layout.addWidget(self.log_console, stretch=1)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.dir_edit.setText(d)

    def _start(self):
        # Apply config and refresh session display
        self.config_panel.apply_to_config()
        import config as cfg

        data_source = getattr(cfg, 'DATA_SOURCE', 'ibl')
        calibration_only = self.chk_calibration_only.isChecked()
        hardware_export = self.chk_hardware_export.isChecked()
        if hardware_export and calibration_only:
            QMessageBox.information(
                self,
                "Choose One Run Mode",
                "Use either Calibration only or Hardware export, not both.",
            )
            self.stage_label.setText("Choose one run mode")
            return
        if data_source == "local":
            if not str(cfg.LOCAL_RECORDING_DIR).strip():
                QMessageBox.information(
                    self,
                    "Local Recording Required",
                    "Select a local recording directory in the Config tab before starting the pipeline.",
                )
                self.stage_label.setText("Set a local recording directory in Config")
                return
            self.session_label.setText(
                f"Source: Local  |  Dir: {cfg.LOCAL_RECORDING_DIR}")
            normal_default_dir = "results_local"
            cali_default_dir = "results_local_cali"
            export_default_dir = "results_local_export"
        else:
            if not str(cfg.IBL_SESSION_EID).strip():
                QMessageBox.information(
                    self,
                    "IBL Session Required",
                    "IBL_SESSION_EID is empty.\n\n"
                    "Choose an IBL session in the Config tab before starting the pipeline.",
                )
                self.stage_label.setText("Set an IBL session EID in Config")
                return
            self.session_label.setText(
                f"Session: {cfg.IBL_SESSION_EID or '<not set>'}  |  "
                f"Probe: {cfg.IBL_PROBE_LABEL}")
            normal_default_dir = "results_ibl"
            cali_default_dir = "results_ibl_cali"
            export_default_dir = "results_ibl_export"

        if hardware_export:
            default_dir = export_default_dir
        elif calibration_only:
            default_dir = cali_default_dir
        else:
            default_dir = normal_default_dir
        current_dir = self.dir_edit.text().strip()
        known_defaults = {
            "results_ibl", "results_local",
            "results_ibl_cali", "results_local_cali",
            "results_ibl_export", "results_local_export",
        }
        output_dir = default_dir if current_dir in known_defaults else (
            current_dir or default_dir)
        self.dir_edit.setText(output_dir)

        self.log_console.clear()
        self._progress_value = 0
        self._last_error_report = ""
        self._captured_output = ""
        self._stop_requested = False
        self.progress_bar.setValue(0)
        self.stage_label.setText("Starting...")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_copy_report.setEnabled(False)

        try:
            fd, snapshot_path = tempfile.mkstemp(
                prefix="bci_config_",
                suffix=".json",
            )
            os.close(fd)
            save_config_snapshot(snapshot_path)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Config Snapshot Error",
                f"Could not prepare the runtime config snapshot:\n\n{e}",
            )
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.stage_label.setText("Config snapshot failed")
            return

        self._config_snapshot_path = snapshot_path
        self._process_output_dir = output_dir
        self._start_time = time.time()
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_process_output)
        self._process.finished.connect(self._on_process_finished)

        args = [
            "-u",
            MAIN_SCRIPT,
            "--config-json", snapshot_path,
            "--source", data_source,
            "--output", output_dir,
        ]
        if calibration_only:
            args.append("--calibration-only")
        if hardware_export:
            args.append("--hardware-export")
        self._process.start(sys.executable, args)
        if not self._process.waitForStarted(3000):
            error_text = self._process.errorString() or "Unknown process start failure"
            self._cleanup_process_artifacts()
            self._process = None
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.stage_label.setText("Failed to start")
            QMessageBox.critical(
                self,
                "Pipeline Start Error",
                f"Could not start the pipeline subprocess:\n\n{error_text}",
            )
            return

        self.status_message.emit(f"Running pipeline ({data_source})...")

    def _stop(self):
        if self._process and self.has_running_worker():
            self._stop_requested = True
            self.stage_label.setText("Stopping...")
            self.status_message.emit("Stopping pipeline...")
            self._process.kill()

    def has_running_worker(self) -> bool:
        return (
            self._process is not None and
            self._process.state() != QProcess.ProcessState.NotRunning
        )

    def close_kills_process_enabled(self) -> bool:
        return self.chk_close_kills_process.isChecked()

    def shutdown_for_close(self, wait_ms: int = 1500) -> bool:
        """Try a graceful stop before window shutdown."""
        if not self.has_running_worker():
            return True
        self._stop_requested = True
        self.stage_label.setText("Stopping for window close...")
        self._process.kill()
        return self._process.waitForFinished(wait_ms)

    def detach_process_for_close(self) -> bool:
        """Let the subprocess continue after the GUI window closes."""
        if not self.has_running_worker():
            return True

        process = self._process
        snapshot_path = self._config_snapshot_path

        for signal, slot in (
            (process.readyReadStandardOutput, self._on_process_output),
            (process.finished, self._on_process_finished),
        ):
            try:
                signal.disconnect(slot)
            except Exception:
                pass

        def _cleanup(*_args):
            if snapshot_path:
                try:
                    os.remove(snapshot_path)
                except OSError:
                    pass
            try:
                BACKGROUND_PROCESSES.remove(process)
            except ValueError:
                pass
            process.deleteLater()

        process.finished.connect(_cleanup)
        process.setParent(None)
        BACKGROUND_PROCESSES.append(process)

        self._process = None
        self._process_output_dir = None
        self._config_snapshot_path = None
        self._stop_requested = False
        return True

    def _append_log_text(self, text: str):
        # Append to console
        cursor = self.log_console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_console.setTextCursor(cursor)
        self.log_console.ensureCursorVisible()

        self._captured_output += text
        if len(self._captured_output) > 1_000_000:
            self._captured_output = self._captured_output[-1_000_000:]

        # Update progress
        self._progress_value = parse_progress(text, self._progress_value)
        self.progress_bar.setValue(self._progress_value)

    def _drain_process_output(self):
        if not self._process:
            return
        data = bytes(self._process.readAllStandardOutput())
        if data:
            self._append_log_text(data.decode("utf-8", errors="replace"))

    def _extract_error_report(self) -> str:
        idx = self._captured_output.rfind(ERROR_REPORT_MARKER)
        if idx >= 0:
            return self._captured_output[idx:].strip()
        return self._captured_output[-12000:].strip()

    def _cleanup_process_artifacts(self):
        if self._config_snapshot_path:
            try:
                os.remove(self._config_snapshot_path)
            except OSError:
                pass
            self._config_snapshot_path = None

    def _on_process_output(self):
        self._drain_process_output()

    def _on_process_finished(self, exit_code: int, _exit_status):
        self._drain_process_output()
        elapsed = time.time() - self._start_time if self._start_time else 0
        output_dir = self._process_output_dir or (self.dir_edit.text().strip() or "")

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._cleanup_process_artifacts()
        self._process = None

        if self._stop_requested:
            self.stage_label.setText(f"Stopped ({elapsed:.1f}s)")
            self.btn_copy_report.setEnabled(False)
            self._last_error_report = ""
            self.status_message.emit(f"Pipeline stopped ({elapsed:.1f}s)")
            self.log_console.appendPlainText("\n[GUI] Pipeline stopped by user.\n")
            self._stop_requested = False
            self._process_output_dir = None
            return

        if exit_code == 0:
            self.progress_bar.setValue(100)
            self.stage_label.setText(f"Complete ({elapsed:.1f}s)")
            self.btn_copy_report.setEnabled(False)
            self._last_error_report = ""
            self.pipeline_finished.emit(output_dir)
            self.status_message.emit(f"Pipeline complete ({elapsed:.1f}s)")
            self._process_output_dir = None
            return

        msg = self._extract_error_report()
        self.stage_label.setText(f"Error ({elapsed:.1f}s)")
        if msg and not self.log_console.toPlainText().rstrip().endswith(msg.rstrip()):
            self.log_console.appendPlainText(f"\n=== ERROR ===\n{msg}\n")
        self._last_error_report = msg
        self.btn_copy_report.setEnabled(bool(msg.strip()))
        self.status_message.emit("Pipeline error: see crash report")

        path_lines = [
            line for line in msg.splitlines()
            if line.startswith("Crash log:")
            or line.startswith("Crash archive:")
            or line.startswith("Output copy:")
            or line.startswith("Output archive:")
        ]
        path_text = "\n".join(path_lines)
        QMessageBox.critical(
            self,
            "Pipeline Error",
            "The pipeline stopped with an error.\n\n"
            "Use 'Copy Crash Report' for the full traceback.\n\n"
            f"{path_text}".rstrip(),
        )
        self._process_output_dir = None

    def _copy_crash_report(self):
        if not self._last_error_report:
            QMessageBox.information(
                self,
                "No Crash Report",
                "No crash report is currently available to copy.",
            )
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(self._last_error_report)
        self.status_message.emit("Crash report copied to clipboard")
