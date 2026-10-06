"""Results viewer: file tree + figure/metrics/text display."""
import json
import os

from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget,
    QTreeWidgetItem, QStackedWidget, QLabel, QScrollArea,
    QPlainTextEdit, QTableWidget, QTableWidgetItem, QPushButton,
    QFileDialog, QHeaderView, Qt, QFileSystemWatcher, QPixmap, QFont,
    get_matplotlib_canvas,
)

import numpy as np
from matplotlib.figure import Figure

FigureCanvasQTAgg, NavigationToolbar2QT = get_matplotlib_canvas()


# ---------- Viewers ----------

class FigureViewer(QWidget):
    """Display a PNG image with scroll and zoom.

    Auto-fits the image to the viewport on load and window resize.
    Ctrl+Wheel allows manual zoom; manual zoom disables auto-fit
    until a new image is loaded.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scale = 1.0
        self._pixmap = None
        self._auto_fit = True  # auto-fit until user manually zooms

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.label)
        self.scroll.setWidgetResizable(False)
        layout.addWidget(self.scroll)

    def load_image(self, path: str):
        self._pixmap = QPixmap(path)
        self._auto_fit = True
        self._fit_to_viewport()

    def _fit_to_viewport(self):
        """Compute scale to fit image within the scroll area viewport."""
        if not self._pixmap or self._pixmap.isNull():
            return
        vp = self.scroll.viewport().size()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return
        sx = vp.width() / pw
        sy = vp.height() / ph
        self._scale = min(sx, sy, 1.0)  # never upscale beyond 1:1
        self._update_display()

    def _update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                int(self._pixmap.width() * self._scale),
                int(self._pixmap.height() * self._scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.label.setPixmap(scaled)
            self.label.adjustSize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self._fit_to_viewport()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._auto_fit = False  # user took manual control
            delta = event.angleDelta().y()
            if delta > 0:
                self._scale *= 1.15
            else:
                self._scale /= 1.15
            self._scale = max(0.1, min(self._scale, 10.0))
            self._update_display()
            event.accept()
        else:
            super().wheelEvent(event)


class TextViewer(QWidget):
    """Display text files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Menlo", 11))
        layout.addWidget(self.text_edit)

    def load_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self.text_edit.setPlainText(content)
        except Exception as e:
            self.text_edit.setPlainText(f"Error reading file: {e}")


class MetricsViewer(QWidget):
    """Display JSON or npy data as a table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.setFont(QFont("Menlo", 11))
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def load_json(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {e}"))
            return

        if isinstance(data, dict):
            self.table.setRowCount(len(data))
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Key", "Value"])
            for i, (k, v) in enumerate(data.items()):
                self.table.setItem(i, 0, QTableWidgetItem(str(k)))
                self.table.setItem(i, 1, QTableWidgetItem(str(v)))
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                keys = list(data[0].keys())
                self.table.setRowCount(len(data))
                self.table.setColumnCount(len(keys))
                self.table.setHorizontalHeaderLabels(keys)
                for i, row in enumerate(data):
                    for j, k in enumerate(keys):
                        self.table.setItem(i, j, QTableWidgetItem(str(row.get(k, ""))))
            else:
                self.table.setRowCount(len(data))
                self.table.setColumnCount(1)
                self.table.setHorizontalHeaderLabels(["Value"])
                for i, v in enumerate(data):
                    self.table.setItem(i, 0, QTableWidgetItem(str(v)))

        self.table.resizeColumnsToContents()

    def load_npy(self, path: str):
        try:
            import numpy as np
            arr = np.load(path, allow_pickle=True)
        except Exception as e:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {e}"))
            return

        # Show summary stats
        info = [
            ("Shape", str(arr.shape)),
            ("Dtype", str(arr.dtype)),
            ("Size", str(arr.size)),
        ]
        if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
            info.extend([
                ("Min", f"{arr.min():.6g}"),
                ("Max", f"{arr.max():.6g}"),
                ("Mean", f"{arr.mean():.6g}"),
                ("Std", f"{arr.std():.6g}"),
            ])
            if arr.ndim == 1 and arr.size <= 200:
                info.append(("Values", str(arr.tolist())))
        elif arr.dtype == bool:
            info.extend([
                ("True count", str(arr.sum())),
                ("False count", str((~arr).sum())),
            ])

        self.table.setRowCount(len(info))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Property", "Value"])
        for i, (k, v) in enumerate(info):
            self.table.setItem(i, 0, QTableWidgetItem(k))
            self.table.setItem(i, 1, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()


class DriftViewer(QWidget):
    """Interactive drift map with matplotlib zoom/pan in Qt."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._fig = Figure(figsize=(12, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    def load_drift_data(self, clustering_dir):
        """Load spike data from clustering/ dir and render interactive drift map."""
        self._fig.clear()

        try:
            times = np.load(os.path.join(clustering_dir, 'spike_times.npy'))
            channels = np.load(os.path.join(clustering_dir, 'spike_channels.npy'))
            amps = np.load(os.path.join(clustering_dir, 'spike_amplitudes.npy'))
        except FileNotFoundError:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Spike data not found in clustering/',
                   ha='center', va='center', transform=ax.transAxes)
            self._canvas.draw()
            return

        import config
        pos_path = os.path.join(clustering_dir, 'channel_positions.npy')
        if os.path.exists(pos_path):
            positions = np.load(pos_path)
        else:
            from data_loader import _default_neuropixels_geometry
            positions = _default_neuropixels_geometry()

        times_s = times.astype(np.float64) / config.SAMPLE_RATE
        depths = positions[np.clip(channels, 0, len(positions) - 1), 1]

        max_pts = 100_000
        n = len(times_s)
        if n > max_pts:
            idx = np.random.default_rng(42).choice(n, max_pts, replace=False)
            idx.sort()
        else:
            idx = np.arange(n)

        sizes = np.clip(amps[idx] * 0.5, 0.5, 4.0)

        ax = self._fig.add_subplot(111)
        ax.scatter(times_s[idx], depths[idx], s=sizes,
                   c=depths[idx], cmap='viridis', alpha=0.2,
                   edgecolors='none', rasterized=True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Depth (um)')
        ax.set_title(f'Interactive Drift Map ({n} spikes)')
        ax.grid(True, alpha=0.2)

        self._fig.tight_layout()
        self._canvas.draw()


# ---------- Results panel ----------

class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_dir = None

        layout = QVBoxLayout(self)

        # Top bar
        top = QHBoxLayout()
        self.dir_label = QLabel("No results loaded")
        btn_open = QPushButton("Open Directory...")
        btn_open.clicked.connect(self._open_directory)
        top.addWidget(self.dir_label, stretch=1)
        top.addWidget(btn_open)
        layout.addLayout(top)

        # Splitter: tree | viewer
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File"])
        self.tree.setMinimumWidth(250)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.splitter.addWidget(self.tree)

        self.stack = QStackedWidget()
        self.figure_viewer = FigureViewer()
        self.text_viewer = TextViewer()
        self.metrics_viewer = MetricsViewer()
        self.drift_viewer = DriftViewer()
        self.placeholder = QLabel("Select a file to view")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.stack.addWidget(self.placeholder)    # 0
        self.stack.addWidget(self.figure_viewer)   # 1
        self.stack.addWidget(self.text_viewer)     # 2
        self.stack.addWidget(self.metrics_viewer)  # 3
        self.stack.addWidget(self.drift_viewer)    # 4
        self.splitter.addWidget(self.stack)

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        layout.addWidget(self.splitter)

        # File watcher for auto-refresh
        self._watcher = QFileSystemWatcher()
        self._watcher.directoryChanged.connect(self._on_dir_changed)

    def _open_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Open Results Directory")
        if d:
            self.load_directory(d)

    def load_directory(self, path: str):
        """Load and display all result files in the given directory."""
        self._current_dir = path
        self.dir_label.setText(path)

        # Stop watching old dirs
        watched = self._watcher.directories()
        if watched:
            self._watcher.removePaths(watched)

        # Watch new dir
        self._watcher.addPath(path)

        self._refresh_tree()

    def _on_dir_changed(self, path):
        self._refresh_tree()

    def _refresh_tree(self):
        if not self._current_dir or not os.path.isdir(self._current_dir):
            return

        self.tree.clear()

        categories = {
            "Figures": [],
            "Data": [],
            "Reports": [],
            "Logs": [],
        }

        for root, dirs, files in os.walk(self._current_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self._current_dir)
                ext = os.path.splitext(fname)[1].lower()

                if ext == ".png":
                    categories["Figures"].append((rel, fpath))
                elif ext in (".npy", ".npz", ".xlsx"):
                    categories["Data"].append((rel, fpath))
                elif ext in (".json", ".txt"):
                    if fname.startswith("log_"):
                        categories["Logs"].append((rel, fpath))
                    else:
                        categories["Reports"].append((rel, fpath))
                else:
                    categories["Data"].append((rel, fpath))

        for cat_name, file_list in categories.items():
            if not file_list:
                continue
            cat_item = QTreeWidgetItem([cat_name])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            for rel, fpath in file_list:
                child = QTreeWidgetItem([rel])
                child.setData(0, Qt.ItemDataRole.UserRole, fpath)
                cat_item.addChild(child)
            self.tree.addTopLevelItem(cat_item)
            cat_item.setExpanded(True)

    def _on_item_clicked(self, item: QTreeWidgetItem, col: int):
        fpath = item.data(0, Qt.ItemDataRole.UserRole)
        if not fpath or not os.path.isfile(fpath):
            return

        ext = os.path.splitext(fpath)[1].lower()

        if ext == ".png":
            fname = os.path.basename(fpath)
            if fname == "drift_map.png":
                clustering_dir = os.path.join(
                    os.path.dirname(fpath), 'clustering')
                if os.path.isdir(clustering_dir):
                    self.drift_viewer.load_drift_data(clustering_dir)
                    self.stack.setCurrentWidget(self.drift_viewer)
                else:
                    self.figure_viewer.load_image(fpath)
                    self.stack.setCurrentWidget(self.figure_viewer)
            else:
                self.figure_viewer.load_image(fpath)
                self.stack.setCurrentWidget(self.figure_viewer)
        elif ext == ".json":
            self.metrics_viewer.load_json(fpath)
            self.stack.setCurrentWidget(self.metrics_viewer)
        elif ext == ".npy":
            self.metrics_viewer.load_npy(fpath)
            self.stack.setCurrentWidget(self.metrics_viewer)
        elif ext in (".txt", ".log"):
            self.text_viewer.load_file(fpath)
            self.stack.setCurrentWidget(self.text_viewer)
        else:
            self.text_viewer.load_file(fpath)
            self.stack.setCurrentWidget(self.text_viewer)
