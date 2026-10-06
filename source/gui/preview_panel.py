"""Data preview panel — heatmap + probe layout visualisation."""

import traceback

import numpy as np
from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QDoubleSpinBox, QSplitter, QCheckBox, QMessageBox,
    QThread, pyqtSignal, Qt, get_matplotlib_canvas,
)

from matplotlib.figure import Figure

FigureCanvasQTAgg, NavigationToolbar2QT = get_matplotlib_canvas()

import config
from preprocessing import apply_car, design_fir_highpass, FIRFilter


# ------------------------------------------------------------------ #
# Worker thread                                                       #
# ------------------------------------------------------------------ #

class PreviewWorker(QThread):
    """Background thread that loads a short data chunk and channel geometry."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, start_s: float = 0.0, duration_s: float = 2.0):
        super().__init__()
        self.start_s = start_s
        self.duration_s = duration_s

    def run(self):
        streamer = None
        try:
            # --- open streamer + load geometry -----------------------
            if config.DATA_SOURCE == "local":
                from local_data_loader import (
                    LocalDataStreamer, load_local_channel_geometry,
                )
                self.progress.emit("Opening local recording...")
                streamer = LocalDataStreamer(
                    config.LOCAL_RECORDING_DIR,
                    ap_folder=config.LOCAL_AP_FOLDER or None,
                )
                streamer.open()
                self.progress.emit("Loading channel geometry...")
                positions, _ = load_local_channel_geometry(
                    config.LOCAL_RECORDING_DIR,
                    ks_settings_path=config.LOCAL_KS_SETTINGS_PATH or None,
                )
            else:
                from data_loader import (
                    connect_one, load_channel_geometry, RawDataStreamer,
                )
                self.progress.emit("Connecting to IBL...")
                one = connect_one()
                eid = config.IBL_SESSION_EID
                probe = config.IBL_PROBE_LABEL
                self.progress.emit("Downloading IBL data (may take a while)...")
                streamer = RawDataStreamer(one, eid, probe)
                streamer.open()
                self.progress.emit("Loading channel geometry...")
                positions, _ = load_channel_geometry(one, eid, probe)

            total_dur = streamer.total_duration_s

            # --- read raw data --------------------------------------
            self.progress.emit("Reading raw data...")
            raw = streamer.read_calibration_data(
                duration_s=self.duration_s,
                start_s=self.start_s,
            )
            n_samples, n_channels = raw.shape
            sample_rate = getattr(streamer, '_sample_rate', None) or config.SAMPLE_RATE

            # --- lightweight filtering (no calibration needed) ------
            self.progress.emit("Filtering...")
            filtered = apply_car(raw.copy())
            coeffs = design_fir_highpass(fs=sample_rate)
            fir = FIRFilter(
                coeffs,
                chunk_samples=n_samples,
                precision_bits=32,
            )  # full precision (0 is falsy)
            fir.reset(n_channels)
            filtered = fir.process_chunk(filtered)

            # --- per-channel RMS ------------------------------------
            rms = np.sqrt(np.mean(filtered ** 2, axis=0))

            self.finished.emit({
                "raw": raw,
                "filtered": filtered,
                "positions": positions,
                "rms": rms,
                "total_duration_s": total_dur,
                "start_s": self.start_s,
                "duration_s": self.duration_s,
                "sample_rate": sample_rate,
                "n_channels": n_channels,
            })

        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            if streamer is not None:
                try:
                    streamer.close()
                except Exception:
                    pass


# ------------------------------------------------------------------ #
# Preview panel widget                                                #
# ------------------------------------------------------------------ #

class PreviewPanel(QWidget):
    """Main preview widget with heatmap, probe map, and stats bar."""

    def __init__(self, config_panel=None, parent=None):
        super().__init__(parent)
        self._config_panel = config_panel
        self._data = None       # dict from worker
        self._worker = None

        self._build_ui()

    # ---- UI construction ----------------------------------------- #

    def _build_ui(self):
        root = QVBoxLayout(self)

        # --- toolbar ------------------------------------------------
        toolbar = QHBoxLayout()

        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self._on_load)
        toolbar.addWidget(self.btn_load)

        toolbar.addWidget(QLabel("View:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Raw", "Filtered"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.combo_mode)

        toolbar.addWidget(QLabel("Start (s):"))
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0, 1e6)
        self.spin_start.setDecimals(2)
        self.spin_start.setValue(0.0)
        self.spin_start.setSingleStep(1.0)
        toolbar.addWidget(self.spin_start)

        toolbar.addWidget(QLabel("Duration (s):"))
        self.spin_dur = QDoubleSpinBox()
        self.spin_dur.setRange(0.1, 60.0)
        self.spin_dur.setDecimals(2)
        self.spin_dur.setValue(2.0)
        self.spin_dur.setSingleStep(0.5)
        toolbar.addWidget(self.spin_dur)

        toolbar.addWidget(QLabel("Gain:"))
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.1, 100.0)
        self.spin_gain.setDecimals(1)
        self.spin_gain.setValue(1.0)
        self.spin_gain.setSingleStep(0.5)
        self.spin_gain.valueChanged.connect(self._on_gain_changed)
        toolbar.addWidget(self.spin_gain)

        self.lbl_info = QLabel("")
        toolbar.addWidget(self.lbl_info)
        toolbar.addStretch()
        root.addLayout(toolbar)

        # --- main area: splitter (heatmap | probe) ------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # left: heatmap
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.fig_heat = Figure(tight_layout=True)
        self.canvas_heat = FigureCanvasQTAgg(self.fig_heat)
        self.nav_heat = NavigationToolbar2QT(self.canvas_heat, left_widget)
        left_layout.addWidget(self.nav_heat)
        left_layout.addWidget(self.canvas_heat)
        splitter.addWidget(left_widget)

        # right: probe
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.fig_probe = Figure(tight_layout=True)
        self.canvas_probe = FigureCanvasQTAgg(self.fig_probe)
        right_layout.addWidget(self.canvas_probe)
        self.chk_aspect = QCheckBox("True aspect ratio")
        self.chk_aspect.setChecked(False)
        self.chk_aspect.stateChanged.connect(self._refresh_probe)
        right_layout.addWidget(self.chk_aspect)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # --- stats bar ----------------------------------------------
        self.lbl_stats = QLabel("")
        root.addWidget(self.lbl_stats)

    # ---- actions -------------------------------------------------- #

    def _on_load(self):
        """Apply config, create worker, start loading."""
        if self._config_panel is not None:
            self._config_panel.apply_to_config()

        if config.DATA_SOURCE == "local" and not str(config.LOCAL_RECORDING_DIR).strip():
            QMessageBox.information(
                self,
                "Local Recording Required",
                "Select a local recording directory in the Config tab before loading a preview.",
            )
            self.lbl_info.setText("Set a local recording directory in Config first")
            return

        if config.DATA_SOURCE == "ibl" and not str(config.IBL_SESSION_EID).strip():
            QMessageBox.information(
                self,
                "IBL Session Required",
                "IBL_SESSION_EID is empty.\n\n"
                "Choose an IBL session in the Config tab before loading a preview.",
            )
            self.lbl_info.setText("Set an IBL session EID in Config first")
            return

        self.btn_load.setEnabled(False)
        self.lbl_info.setText("Loading...")

        self._worker = PreviewWorker(
            start_s=self.spin_start.value(),
            duration_s=self.spin_dur.value(),
        )
        self._worker.finished.connect(self._on_worker_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.start()

    def _on_worker_done(self, result: dict):
        self._data = result
        self._worker = None
        self.btn_load.setEnabled(True)
        n_ch = result["n_channels"]
        sr = result["sample_rate"]
        total = result["total_duration_s"]
        self.lbl_info.setText(
            f"{n_ch} ch | {sr} Hz | total {total:.1f} s"
        )
        self._refresh_heatmap()
        self._refresh_probe()
        self._refresh_stats()

    def _on_worker_progress(self, msg: str):
        self.lbl_info.setText(msg)

    def _on_worker_error(self, msg: str):
        self._worker = None
        self.btn_load.setEnabled(True)
        last_line = msg.splitlines()[-1] if msg else "Unknown error"
        self.lbl_info.setText(f"Error: {last_line}")
        self.lbl_stats.setText(msg.strip()[-300:] if msg else "")

    def has_running_worker(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # ---- drawing -------------------------------------------------- #

    def _current_matrix(self):
        """Return the selected data matrix (channels x time)."""
        if self._data is None:
            return None
        if self.combo_mode.currentText() == "Filtered":
            return self._data["filtered"]
        return self._data["raw"]

    def _refresh_heatmap(self):
        """Draw channel x time imshow with RdBu_r colormap."""
        mat = self._current_matrix()
        if mat is None:
            return

        self.fig_heat.clear()
        ax = self.fig_heat.add_subplot(111)

        gain = self.spin_gain.value()
        abs_max = np.percentile(np.abs(mat), 99)
        vmax = abs_max / gain if abs_max > 0 else 1.0

        start_s = self._data["start_s"]
        duration_s = self._data["duration_s"]
        n_channels = mat.shape[1]

        # imshow expects (rows, cols) — rows = channels, cols = time
        ax.imshow(
            mat.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[start_s, start_s + duration_s, 0, n_channels],
            interpolation="nearest",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel")
        ax.set_title(self.combo_mode.currentText())
        self.canvas_heat.draw_idle()

    def _refresh_probe(self):
        """Draw probe layout scatter — green = normal, red = noisy (top 5% RMS)."""
        if self._data is None:
            return

        positions = self._data["positions"]
        rms = self._data["rms"]

        self.fig_probe.clear()
        ax = self.fig_probe.add_subplot(111)

        n_ch = min(len(rms), positions.shape[0])
        x = positions[:n_ch, 0]
        y = positions[:n_ch, 1]
        ch_rms = rms[:n_ch]

        threshold = np.percentile(ch_rms, 95)
        noisy = ch_rms >= threshold
        normal = ~noisy

        ax.scatter(x[normal], y[normal], s=8, c="green", label="Normal", zorder=2)
        ax.scatter(x[noisy], y[noisy], s=8, c="red", label="Noisy", zorder=3)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title("Probe Layout")
        ax.legend(loc="upper right", fontsize=7)

        if self.chk_aspect.isChecked():
            ax.set_aspect("equal")
        else:
            ax.set_aspect("auto")

        self.canvas_probe.draw_idle()

    def _refresh_stats(self):
        """Update the stats bar label."""
        if self._data is None:
            return

        rms = self._data["rms"]
        mat = self._current_matrix()
        n_ch = self._data["n_channels"]
        duration = self._data["duration_s"]

        mean_rms = float(np.mean(rms))
        peak = float(np.max(np.abs(mat)))

        threshold = np.percentile(rms, 95)
        n_noisy = int(np.sum(rms >= threshold))
        n_active = n_ch - n_noisy

        self.lbl_stats.setText(
            f"RMS: {mean_rms:.1f} uV  |  Peak: {peak:.1f} uV  |  "
            f"Active: {n_active}  |  Noisy: {n_noisy}  |  "
            f"Duration: {duration:.2f} s"
        )

    # ---- slots ----------------------------------------------------- #

    def _on_mode_changed(self):
        """Toggle between raw and filtered data views."""
        self._refresh_heatmap()
        self._refresh_stats()

    def _on_gain_changed(self):
        """Re-draw heatmap when gain spinner changes."""
        self._refresh_heatmap()
