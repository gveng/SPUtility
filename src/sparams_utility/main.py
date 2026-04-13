from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from sparams_utility.models.state import AppState
from sparams_utility.ui.main_window import MainWindow


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("S-Parameters Utility")

    pg.setConfigOptions(antialias=True)

    state = AppState()
    window = MainWindow(state)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
