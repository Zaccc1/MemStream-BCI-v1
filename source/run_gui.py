"""Launch the BCI Neural Processing Pipeline GUI."""
import sys
import os
import traceback

# Ensure the project root is on sys.path so 'import config' etc. work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runtime_support import format_error_report, write_crash_report


def main():
    try:
        from gui.qt_compat import QApplication, QMessageBox
        from gui.app import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("BCI Neural Processing Pipeline")
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception:
        report = format_error_report(
            "GUI startup crash",
            traceback_text=traceback.format_exc(),
        )
        paths = write_crash_report(report)
        path_lines = []
        if paths.get("global_path"):
            path_lines.append(f"Crash log: {paths['global_path']}")
        if paths.get("global_archive_path"):
            path_lines.append(f"Crash archive: {paths['global_archive_path']}")
        for err in paths.get("write_errors", []):
            path_lines.append(f"Crash report warning: {err}")
        display_report = report.rstrip()
        if path_lines:
            display_report += "\n" + "\n".join(path_lines) + "\n"
        print(display_report, file=sys.stderr)

        try:
            from gui.qt_compat import QApplication, QMessageBox
            app = QApplication.instance()
            owns_app = False
            if app is None:
                app = QApplication(sys.argv)
                owns_app = True
            QMessageBox.critical(
                None,
                "BCI Pipeline Crashed",
                "The GUI failed to start.\n\n"
                "Crash report location:\n"
                f"{chr(10).join(path_lines) if path_lines else 'Crash report could not be written.'}",
            )
            if owns_app:
                app.quit()
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    sys.exit(main())
