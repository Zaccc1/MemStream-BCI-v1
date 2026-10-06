"""Qt compatibility layer — auto-selects PyQt6 or PyQt5."""

try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from PyQt6.QtCore import pyqtSignal, Qt
    from PyQt6.QtGui import QFont, QPixmap, QTextCursor
    from PyQt6.QtWidgets import *
    from PyQt6.QtWidgets import QMessageBox
    from PyQt6.QtCore import QThread, QFileSystemWatcher, QProcess
    QT_VERSION = 6
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import pyqtSignal, Qt
    from PyQt5.QtGui import QFont, QPixmap, QTextCursor
    from PyQt5.QtWidgets import *
    from PyQt5.QtWidgets import QMessageBox
    from PyQt5.QtCore import QThread, QFileSystemWatcher, QProcess
    QT_VERSION = 5

# PyQt6 moved some enums to scoped form; PyQt5 uses flat form.
# Provide aliases so code written for PyQt6 works on PyQt5.
if QT_VERSION == 5:
    # Qt.AlignmentFlag.AlignCenter -> Qt.AlignCenter
    if not hasattr(Qt, 'AlignmentFlag'):
        class _AlignmentFlag:
            AlignCenter = Qt.AlignCenter
            AlignRight = Qt.AlignRight
            AlignLeft = Qt.AlignLeft
        Qt.AlignmentFlag = _AlignmentFlag

    # Qt.Orientation.Horizontal -> Qt.Horizontal
    if not hasattr(Qt, 'Orientation'):
        class _Orientation:
            Horizontal = Qt.Horizontal
            Vertical = Qt.Vertical
        Qt.Orientation = _Orientation

    # Qt.ItemFlag.ItemIsSelectable -> Qt.ItemIsSelectable
    if not hasattr(Qt, 'ItemFlag'):
        class _ItemFlag:
            ItemIsSelectable = Qt.ItemIsSelectable
        Qt.ItemFlag = _ItemFlag

    # Qt.ItemDataRole.UserRole -> Qt.UserRole
    if not hasattr(Qt, 'ItemDataRole'):
        class _ItemDataRole:
            UserRole = Qt.UserRole
        Qt.ItemDataRole = _ItemDataRole

    # Qt.KeyboardModifier.ControlModifier -> Qt.ControlModifier
    if not hasattr(Qt, 'KeyboardModifier'):
        class _KeyboardModifier:
            ControlModifier = Qt.ControlModifier
        Qt.KeyboardModifier = _KeyboardModifier

    # Qt.AspectRatioMode / TransformationMode
    if not hasattr(Qt, 'AspectRatioMode'):
        class _AspectRatioMode:
            KeepAspectRatio = Qt.KeepAspectRatio
        Qt.AspectRatioMode = _AspectRatioMode

    if not hasattr(Qt, 'TransformationMode'):
        class _TransformationMode:
            SmoothTransformation = Qt.SmoothTransformation
        Qt.TransformationMode = _TransformationMode

    # QTextCursor.MoveOperation -> QTextCursor enum directly in PyQt5
    if not hasattr(QTextCursor, 'MoveOperation'):
        class _MoveOperation:
            End = QTextCursor.End
        QTextCursor.MoveOperation = _MoveOperation

    if not hasattr(QProcess, 'ProcessChannelMode'):
        class _ProcessChannelMode:
            MergedChannels = QProcess.MergedChannels
            SeparateChannels = QProcess.SeparateChannels
        QProcess.ProcessChannelMode = _ProcessChannelMode

    if not hasattr(QProcess, 'ProcessState'):
        class _ProcessState:
            NotRunning = QProcess.NotRunning
            Starting = QProcess.Starting
            Running = QProcess.Running
        QProcess.ProcessState = _ProcessState

    if not hasattr(QMessageBox, 'StandardButton'):
        class _StandardButton:
            Yes = QMessageBox.Yes
            No = QMessageBox.No
            Ok = QMessageBox.Ok
            Cancel = QMessageBox.Cancel
        QMessageBox.StandardButton = _StandardButton

# matplotlib backend
if QT_VERSION == 6:
    MPL_BACKEND_MODULE = 'matplotlib.backends.backend_qtagg'
else:
    MPL_BACKEND_MODULE = 'matplotlib.backends.backend_qt5agg'

def get_matplotlib_canvas():
    """Return (FigureCanvasQTAgg, NavigationToolbar2QT) classes."""
    if QT_VERSION == 6:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    else:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
    return FigureCanvasQTAgg, NavigationToolbar2QT
