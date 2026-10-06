"""Main application window with four tabs: Config, Preview, Run, Results."""

from gui.qt_compat import (
    QMainWindow, QTabWidget, QStatusBar, QMessageBox,
)

from gui.config_editor import ConfigPanel
from gui.pipeline_runner import RunPanel
from gui.preview_panel import PreviewPanel
from gui.results_viewer import ResultsPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Neural Processing Pipeline")
        self.resize(1200, 800)

        # --- Tabs ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.config_panel = ConfigPanel()
        self.preview_panel = PreviewPanel(self.config_panel)
        self.run_panel = RunPanel(self.config_panel)
        self.results_panel = ResultsPanel()

        self.tabs.addTab(self.config_panel, "Config")
        self.tabs.addTab(self.preview_panel, "Preview")
        self.tabs.addTab(self.run_panel, "Run")
        self.tabs.addTab(self.results_panel, "Results")

        # --- Status bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # --- Connections ---
        self.run_panel.pipeline_finished.connect(self._on_pipeline_finished)
        self.run_panel.status_message.connect(self.status_bar.showMessage)

    def _on_pipeline_finished(self, output_dir: str):
        """Switch to Results tab and load the output directory."""
        self.results_panel.load_directory(output_dir)
        self.tabs.setCurrentWidget(self.results_panel)
        self.status_bar.showMessage(f"Pipeline complete - {output_dir}")

    def closeEvent(self, event):
        """Coordinate preview and pipeline shutdown before closing."""
        if self.preview_panel.has_running_worker():
            QMessageBox.warning(
                self,
                "Preview Still Loading",
                "The Preview tab is still loading data.\n\n"
                "Please wait for the preview to finish before closing the window.",
            )
            event.ignore()
            return

        if self.run_panel.has_running_worker():
            if self.run_panel.close_kills_process_enabled():
                self.status_bar.showMessage("Stopping background work before exit...")
                stopped = self.run_panel.shutdown_for_close(wait_ms=1500)
                if not stopped:
                    QMessageBox.warning(
                        self,
                        "Background Task Still Running",
                        "A background task is still busy in a blocking step.\n\n"
                        "To avoid losing cleanup handlers and crash information, "
                        "the window will stay open for now.\n\n"
                        "If you want to close the window while work continues in "
                        "the background, uncheck 'Close window ends background "
                        "Python process' in the Run tab and close again.",
                )
                    event.ignore()
                    return
            else:
                reply = QMessageBox.question(
                    self,
                    "Background Task Still Running",
                    "A background task is still running in the background.\n\n"
                    "Close the window anyway and leave the Python process alive?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                self.run_panel.detach_process_for_close()
        super().closeEvent(event)
