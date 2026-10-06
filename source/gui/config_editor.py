"""Config parameter editing panel with categorized, scrollable layout."""
import json
import copy
from pathlib import Path

from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QComboBox, QPushButton, QFileDialog, QFormLayout, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, Qt,
)

import config

# ---------- Category definitions ----------
# Each tuple: (display_name, list_of_param_names)
CATEGORIES = [
    ("Data Source", [
        "DATA_SOURCE",
    ]),
    ("IBL Data Source", [
        "IBL_BASE_URL", "IBL_SESSION_EID", "IBL_PROBE_LABEL", "IBL_DATASET_TYPE",
    ]),
    ("Local Data Source", [
        "LOCAL_RECORDING_DIR", "LOCAL_AP_FOLDER",
        "LOCAL_KS_SETTINGS_PATH", "LOCAL_REFERENCE_PATH",
        "LOCAL_TTL_DIR", "LOCAL_SKIP_BRAIN_REGIONS",
        "LOCAL_PREPROCESSING_MODE",
    ]),
    ("Raw Data", [
        "N_CHANNELS", "SAMPLE_RATE", "DTYPE_RAW", "UV_PER_BIT",
    ]),
    ("Calibration", [
        "CALIBRATION_DURATION_S", "N_CALIB_CHUNKS",
        "CALI_CACHE_ENABLE", "CALI_CACHE_SUBDIR", "CALI_CACHE_VERSION",
    ]),
    ("Signal Processing", [
        "CAR_METHOD",
        "FIR_NUM_TAPS", "FIR_HIGHPASS_FREQ", "FIR_WINDOW",
        "WHITEN_MODE", "WHITEN_NRANGE", "WHITEN_OVERLAP", "WHITEN_EPSILON",
    ]),
    ("Precision (bits)", [
        "PRECISION_ADC", "PRECISION_DAC",
        "PRECISION_FIR", "PRECISION_WHITEN", "PRECISION_MATCH",
        "PRECISION_SA", "PRECISION_PCA", "PRECISION_ASSIGN", "PRECISION_LDA",
    ]),
    ("Quantization", [
        "QUANTIZATION_BACKEND", "QUANT_DIAGNOSTICS",
        "QUANT_DIAG_MAX_PRINTS_PER_OP", "QUANT_CLIP_WARN_FRAC",
        "QUANT_RANGE_PERCENTILE", "QUANT_RANGE_SAMPLE_LIMIT",
        "QUANT_RANGE_SAMPLE_PER_UPDATE", "QUANT_RANGE_SYMMETRIC",
        "QUANT_CALIBRATION_SUBDIR", "QUANT_REFIT_AFTER_RANGE_FREEZE",
        "QUANT_RANGE_STAGE_OVERRIDES", "QUANT_RANGE_OVERRIDE_PATH",
    ]),
    ("Streaming", [
        "CHUNK_DURATION_MS", "MAX_TEST_BATCHES",
    ]),
    ("Hardware Export", [
        "HARDWARE_EXPORT_ENABLE", "HARDWARE_EXPORT_SUBDIR",
        "HARDWARE_EXPORT_BATCH_INDEX", "HARDWARE_EXPORT_STAGES",
        "HARDWARE_EXPORT_MAX_VMM_ROWS", "HARDWARE_EXPORT_ONNX",
        "HARDWARE_EXPORT_PLOTS", "HARDWARE_EXPORT_SPLIT_MODELS",
        "HARDWARE_EXPORT_PLOT_MAX_POINTS",
    ]),
    ("Template Matching", [
        "MATCH_NT", "MATCH_N_TEMPLATES", "MATCH_STRIDE",
        "MATCH_THRESHOLD", "MATCH_NMS_SAMPLES", "MATCH_TH_SINGLE_CH",
        "TEMPLATE_FROM_DATA", "TEMPLATE_TH_DETECT",
        "TEMPLATE_ISOLATION_T", "TEMPLATE_ISOLATION_CH", "TEMPLATE_MAX_CLIPS",
    ]),
    ("Spatial Aggregation", [
        "SA_NEAREST_CHANNELS", "SA_N_SCALES", "SA_SIGMA_UM",
        "SA_COARSE_THRESHOLD", "SA_FINAL_THRESHOLD",
        "SA_NMS_SAMPLES", "SA_NMS_CHANNELS", "SA_MODE", "SA_DENSE",
    ]),
    ("Clustering", [
        "CLUST_N_NEARBY", "CLUST_SNIPPET_HW", "CLUST_MIN_COUNT",
        "CLUST_TEMPORAL_PCA_NC", "CLUST_KMEANS_CLEAN_K", "CLUST_KMEANS_CLEAN_MIN",
        "CLUST_WAVEFORM_MAX_CLIPS",
        "CLUST_LDA_MODE", "CLUST_LDA_N_COMPONENTS",
        "CLUST_LDA_MIN_SAMPLES_PER_CLASS", "CLUST_LDA_DIRECT_N_CHANNELS",
        "CLUST_REGION_SIZE", "CLUST_REGION_OVERLAP",
        "CLUST_ASSIGN_REJECT_ENABLE", "CLUST_ASSIGN_REJECT_MIN_SAMPLES",
        "CLUST_ASSIGN_REJECT_QUANTILE", "CLUST_ASSIGN_REJECT_SCALE",
        "CLUST_ASSIGN_MARGIN_MIN",
        "CLUST_ENABLE_CENTER_UPDATES", "CLUST_UPDATE_HIGHCONF_ONLY",
        "CLUST_UPDATE_QUANTILE", "CLUST_UPDATE_SCALE", "CLUST_UPDATE_MARGIN_MIN",
        "KMEANS_CLEAN_MAX_ITER", "KMEANS_CONVERGENCE_TOL",
    ]),
    ("Merging", [
        "MERGE_SIMILARITY_THRESHOLD", "MERGE_ACG_THRESHOLD",
        "MERGE_CCG_THRESHOLD", "MERGE_SPATIAL_SIGMA", "MERGE_MIN_SPIKES",
        "CCG_TBIN", "CCG_NBINS", "CCG_R12_ACG_THRESHOLD", "CCG_R12_CCG_THRESHOLD",
    ]),
    ("Decoding", [
        "DECODE_VAL_DURATION_S", "DECODE_ACC_THRESHOLD",
        "DECODE_CHOICE_WINDOW", "DECODE_FEEDBACK_WINDOW",
        "DECODE_N_SPLITS", "DECODE_N_SHUFFLE", "DECODE_QUANT_BITS",
    ]),
    ("KS4 Integration", [
        "KS4_IMPORT_MODE", "KS4_SOURCE_DIR", "KS4_RESULTS_SUBDIR",
        "KS4_KEEP_CALI_BINARY", "KS4_SAVE_EXTRA_VARS",
        "KS4_DEVICE", "KS4_RUN_IN_SUBPROCESS", "KS4_THREAD_LIMIT",
        "KS4_SUBPROCESS_TIMEOUT_S", "KS4_BATCH_SIZE", "KS4_NBLOCKS",
        "KS4_NT", "KS4_N_PCS",
        "KS4_TH_UNIVERSAL", "KS4_TH_LEARNED",
        "KS4_TMIN", "KS4_TMAX", "KS4_ARTIFACT_THRESHOLD",
        "KS4_DO_CAR", "KS4_INVERT_SIGN",
        "CALI_CENTER_MATCH_MAX_DT_SAMPLES",
        "KS4_CACHE_ENABLE", "KS4_CACHE_SUBDIR",
        "KS4_FULL_ENABLE", "KS4_FULL_CACHE_SUBDIR",
        "KS4_FULL_DEVICE", "KS4_FULL_KEEP_BINARY",
    ]),
    ("Visualization & Diagnostics", [
        "VIS_CHANNELS", "VIS_DURATION_MS",
        "DIAG_DETECTION_AUDIT", "DIAG_FN_ANALYSIS",
        "DIAG_FEATURE_SEPARABILITY", "DIAG_LABEL_ANALYSIS", "DIAG_M1_ONLY",
    ]),
    ("Miscellaneous", [
        "RANDOM_SEED",
    ]),
]

# Known enum choices for string parameters
ENUM_CHOICES = {
    "DATA_SOURCE": ["ibl", "local"],
    "CAR_METHOD": ["median", "mean"],
    "WHITEN_MODE": ["overlap", "non_overlap", "overlap_symmetric", "sliding"],
    "CLUST_LDA_MODE": ["off", "post_pca", "direct", "per_region"],
    "SA_MODE": ["gaussian"],
    "KS4_IMPORT_MODE": ["package", "package_or_source", "source"],
    "FIR_WINDOW": ["hamming", "hann", "blackman", "kaiser"],
    "KS4_DEVICE": ["auto", "cuda", "cpu"],
    "KS4_FULL_DEVICE": ["auto", "cuda", "cpu"],
    "DTYPE_RAW": ["int16", "float32"],
    "QUANTIZATION_BACKEND": ["uint8_int_accum", "fake", "int_accum"],
    "LOCAL_PREPROCESSING_MODE": ["auto", "force_raw", "skip_all"],
}

# Derived parameters that are recomputed from others
DERIVED_PARAMS = {"FIR_GROUP_DELAY", "CHUNK_SAMPLES", "CLUST_WAVEFORM_COLLECT_N_NEARBY"}


def _snapshot_defaults():
    """Capture the original config values as defaults."""
    defaults = {}
    for key in dir(config):
        if key.startswith("_") or not key[0].isupper():
            continue
        val = getattr(config, key)
        if callable(val):
            continue
        defaults[key] = copy.deepcopy(val)
    return defaults


_DEFAULTS = _snapshot_defaults()


QUANT_RANGE_STAGE_ROWS = [
    "FIR.input", "FIR.adc",
    "WHITEN.input", "WHITEN.adc",
    "MATCH.input", "MATCH.adc",
    "SA.sparse.input", "SA.sparse.adc",
    "SA.dense.input", "SA.dense.adc",
    "TPCA.input", "TPCA.adc",
    "ASSIGN.input", "ASSIGN.adc",
    "LDA.input", "LDA.adc",
]


class QuantRangeOverridesEditor(QWidget):
    """Table editor for per-stage quantization range overrides."""

    MODE_AUTO = "Auto"
    MODE_PERCENTILE = "Percentile"
    MODE_RANGE = "Range"

    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self._row_widgets = []
        self._extra_overrides = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Stage", "Mode", "Percentile", "Min", "Max", "Symmetric",
        ])
        self.table.setMinimumHeight(420)
        try:
            h_policy = QSizePolicy.Policy.Expanding
            v_policy = QSizePolicy.Policy.MinimumExpanding
        except AttributeError:
            h_policy = QSizePolicy.Expanding
            v_policy = QSizePolicy.MinimumExpanding
        self.table.setSizePolicy(h_policy, v_policy)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        try:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for col in range(1, 6):
                header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        except AttributeError:
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            for col in range(1, 6):
                header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        self.set_value(value or {})

    def _stage_list(self, overrides):
        stages = list(QUANT_RANGE_STAGE_ROWS)
        for stage in sorted((overrides or {}).keys()):
            if stage not in stages:
                stages.append(stage)
        return stages

    @staticmethod
    def _range_from_spec(spec):
        if isinstance(spec, dict):
            if "range" in spec:
                spec = spec["range"]
            elif "min" in spec and "max" in spec:
                spec = (spec["min"], spec["max"])
            else:
                return None
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            try:
                lo = float(spec[0])
                hi = float(spec[1])
            except (TypeError, ValueError):
                return None
            if hi > lo:
                return lo, hi
        return None

    def _add_row(self, row, stage, spec):
        item = QTableWidgetItem(stage)
        self.table.setItem(row, 0, item)

        mode = QComboBox()
        mode.addItems([self.MODE_AUTO, self.MODE_PERCENTILE, self.MODE_RANGE])
        pct = QDoubleSpinBox()
        pct.setRange(0.0, 100.0)
        pct.setDecimals(3)
        pct.setSingleStep(0.1)
        pct.setValue(float(getattr(config, "QUANT_RANGE_PERCENTILE", 99.0)))
        min_box = QDoubleSpinBox()
        max_box = QDoubleSpinBox()
        for box in (min_box, max_box):
            box.setRange(-1e12, 1e12)
            box.setDecimals(6)
            box.setSingleStep(0.1)
        min_box.setValue(-1.0)
        max_box.setValue(1.0)
        symmetric = QCheckBox()

        range_pair = self._range_from_spec(spec)
        if range_pair is not None:
            mode.setCurrentText(self.MODE_RANGE)
            min_box.setValue(range_pair[0])
            max_box.setValue(range_pair[1])
        elif isinstance(spec, dict) and "percentile" in spec:
            mode.setCurrentText(self.MODE_PERCENTILE)
            pct.setValue(float(spec.get("percentile", pct.value())))
            symmetric.setChecked(bool(spec.get("symmetric", False)))
        else:
            mode.setCurrentText(self.MODE_AUTO)

        widgets = {
            "stage": item,
            "mode": mode,
            "percentile": pct,
            "min": min_box,
            "max": max_box,
            "symmetric": symmetric,
        }
        self._row_widgets.append(widgets)

        self.table.setCellWidget(row, 1, mode)
        self.table.setCellWidget(row, 2, pct)
        self.table.setCellWidget(row, 3, min_box)
        self.table.setCellWidget(row, 4, max_box)
        self.table.setCellWidget(row, 5, symmetric)
        mode.currentTextChanged.connect(
            lambda _text, row_widgets=widgets: self._sync_row_enabled(row_widgets))
        self._sync_row_enabled(widgets)

    def _sync_row_enabled(self, widgets):
        mode = widgets["mode"].currentText()
        is_pct = mode == self.MODE_PERCENTILE
        is_range = mode == self.MODE_RANGE
        widgets["percentile"].setEnabled(is_pct)
        widgets["symmetric"].setEnabled(is_pct)
        widgets["min"].setEnabled(is_range)
        widgets["max"].setEnabled(is_range)

    def set_value(self, value):
        overrides = value if isinstance(value, dict) else {}
        self._row_widgets = []
        stages = self._stage_list(overrides)
        self.table.setRowCount(len(stages))
        for row, stage in enumerate(stages):
            self._add_row(row, stage, overrides.get(stage, {}))

    def value(self):
        overrides = {}
        for widgets in self._row_widgets:
            stage = widgets["stage"].text().strip()
            if not stage:
                continue
            mode = widgets["mode"].currentText()
            if mode == self.MODE_PERCENTILE:
                spec = {"percentile": float(widgets["percentile"].value())}
                if widgets["symmetric"].isChecked():
                    spec["symmetric"] = True
                overrides[stage] = spec
            elif mode == self.MODE_RANGE:
                lo = float(widgets["min"].value())
                hi = float(widgets["max"].value())
                if hi > lo:
                    overrides[stage] = {"min": lo, "max": hi}
        return overrides


class ConfigPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: dict[str, QWidget] = {}  # param_name -> widget
        self._groups: dict[str, QGroupBox] = {}  # category_name -> group box

        layout = QVBoxLayout(self)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for cat_name, param_names in CATEGORIES:
            group = QGroupBox(cat_name)
            group.setCheckable(True)
            group.setChecked(True)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            for name in param_names:
                if name in DERIVED_PARAMS:
                    continue
                if not hasattr(config, name):
                    continue
                val = getattr(config, name)
                widget = self._make_widget(name, val)
                self._widgets[name] = widget
                form.addRow(name, widget)
            group.setLayout(form)
            scroll_layout.addWidget(group)
            self._groups[cat_name] = group

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Connect DATA_SOURCE combo to toggle IBL / Local visibility
        if "DATA_SOURCE" in self._widgets:
            combo = self._widgets["DATA_SOURCE"]
            if isinstance(combo, QComboBox):
                combo.currentTextChanged.connect(self._on_data_source_changed)
                self._on_data_source_changed(combo.currentText())

        # Buttons
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset Defaults")
        btn_save = QPushButton("Save Preset")
        btn_load = QPushButton("Load Preset")
        btn_reset.clicked.connect(self.reset_defaults)
        btn_save.clicked.connect(self.save_preset)
        btn_load.clicked.connect(self.load_preset)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_load)
        layout.addLayout(btn_row)

    # ---------- Data source toggle ----------

    def _on_data_source_changed(self, text):
        """Show/hide IBL vs Local parameter groups based on selection."""
        is_local = (text == "local")
        if "IBL Data Source" in self._groups:
            self._groups["IBL Data Source"].setVisible(not is_local)
        if "Local Data Source" in self._groups:
            self._groups["Local Data Source"].setVisible(is_local)

    # ---------- Widget creation ----------

    # Parameters that get a file/dir Browse button
    _PATH_PARAMS = {
        "LOCAL_RECORDING_DIR", "LOCAL_TTL_DIR",
    }
    _FILE_PARAMS = {
        "LOCAL_KS_SETTINGS_PATH", "LOCAL_REFERENCE_PATH",
        "QUANT_RANGE_OVERRIDE_PATH",
    }

    def _make_widget(self, name, val):
        if name == "QUANT_RANGE_STAGE_OVERRIDES":
            return QuantRangeOverridesEditor(val)

        # Path/file parameters get a QLineEdit + Browse button
        if name in self._PATH_PARAMS or name in self._FILE_PARAMS:
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            line_edit = QLineEdit(str(val) if val else "")
            btn = QPushButton("Browse...")
            if name in self._PATH_PARAMS:
                btn.clicked.connect(
                    lambda checked, le=line_edit: self._browse_dir(le))
            else:
                btn.clicked.connect(
                    lambda checked, le=line_edit: self._browse_file(le))
            row.addWidget(line_edit, stretch=1)
            row.addWidget(btn)
            return container

        if name in ENUM_CHOICES:
            w = QComboBox()
            choices = ENUM_CHOICES[name]
            w.addItems(choices)
            if isinstance(val, str) and val in choices:
                w.setCurrentText(val)
            return w

        if isinstance(val, bool):
            w = QCheckBox()
            w.setChecked(val)
            return w

        if isinstance(val, int):
            w = QSpinBox()
            w.setRange(-999_999_999, 999_999_999)
            w.setValue(val)
            return w

        if isinstance(val, float):
            if val == float("inf"):
                w = QLineEdit(str(val))
                return w
            w = QDoubleSpinBox()
            w.setRange(-1e12, 1e12)
            w.setDecimals(6)
            w.setValue(val)
            return w

        if isinstance(val, (list, tuple, dict)):
            w = QLineEdit(json.dumps(val))
            return w

        # str, None, or anything else
        w = QLineEdit("" if val is None else str(val))
        return w

    # ---------- Browse helpers ----------

    def _browse_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory",
                                              line_edit.text())
        if d:
            line_edit.setText(d)

    def _browse_file(self, line_edit):
        f, _ = QFileDialog.getOpenFileName(self, "Select File",
                                            line_edit.text(),
                                            "All Files (*)")
        if f:
            line_edit.setText(f)

    @staticmethod
    def _get_line_edit_from_container(widget):
        """Extract the QLineEdit from a path-browse container widget."""
        layout = widget.layout()
        if layout is not None:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QLineEdit):
                    return item.widget()
        return None

    # ---------- Read values from widgets ----------

    def _get_value(self, name):
        widget = self._widgets[name]
        default = _DEFAULTS.get(name)

        # Handle path-browse container widgets
        if isinstance(widget, QuantRangeOverridesEditor):
            return widget.value()

        if name in self._PATH_PARAMS or name in self._FILE_PARAMS:
            le = self._get_line_edit_from_container(widget)
            return le.text().strip() if le else (default or "")

        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            # Try to parse as the original type
            if isinstance(default, (list, tuple, dict)):
                try:
                    parsed = json.loads(text)
                    if isinstance(default, dict):
                        return parsed if isinstance(parsed, dict) else default
                    return type(default)(parsed) if isinstance(default, tuple) else parsed
                except (json.JSONDecodeError, TypeError):
                    return default
            if default is None:
                return None if text in ("", "None") else text
            if isinstance(default, float):
                try:
                    return float(text)
                except ValueError:
                    return default
            return text
        return default

    # ---------- Apply to config module ----------

    def apply_to_config(self):
        """Write all widget values into the config module."""
        for name in self._widgets:
            val = self._get_value(name)
            setattr(config, name, val)
        # Recompute derived values
        config.FIR_GROUP_DELAY = (config.FIR_NUM_TAPS - 1) // 2
        config.CHUNK_SAMPLES = int(config.SAMPLE_RATE * config.CHUNK_DURATION_MS / 1000.0)
        if hasattr(config, "CLUST_WAVEFORM_COLLECT_N_NEARBY"):
            config.CLUST_WAVEFORM_COLLECT_N_NEARBY = config.CLUST_N_NEARBY

    # ---------- Reset / Preset ----------

    def reset_defaults(self):
        for name, widget in self._widgets.items():
            val = _DEFAULTS.get(name)
            self._set_widget_value(widget, name, val)

    def _set_widget_value(self, widget, name, val):
        # Handle path-browse container widgets
        if isinstance(widget, QuantRangeOverridesEditor):
            widget.set_value(val if isinstance(val, dict) else {})
            return

        if name in self._PATH_PARAMS or name in self._FILE_PARAMS:
            le = self._get_line_edit_from_container(widget)
            if le:
                le.setText(str(val) if val else "")
            return
        if isinstance(widget, QComboBox):
            idx = widget.findText(str(val))
            if idx >= 0:
                widget.setCurrentIndex(idx)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(val))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(val))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(val))
        elif isinstance(widget, QLineEdit):
            if isinstance(val, (list, tuple, dict)):
                widget.setText(json.dumps(val))
            elif val is None:
                widget.setText("None")
            else:
                widget.setText(str(val))

    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", "", "JSON Files (*.json)"
        )
        if not path:
            return
        data = {name: self._get_value(name) for name in self._widgets}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))
            return
        for name, val in data.items():
            if name == "PRECISION_KMEANS" and "PRECISION_ASSIGN" in self._widgets:
                name = "PRECISION_ASSIGN"
            if name in self._widgets:
                # Restore original type for tuples
                default = _DEFAULTS.get(name)
                if isinstance(default, tuple) and isinstance(val, list):
                    val = tuple(val)
                self._set_widget_value(self._widgets[name], name, val)
