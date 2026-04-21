from __future__ import annotations

import sys
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from sparams_utility.models.state import AppState
from sparams_utility.ui.main_window import MainWindow


def _resource_path(*parts: str) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parents[2]
    return base.joinpath(*parts)


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("S-Parameters Utility")

    icon_path = _resource_path("Images", "Icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    splash = None
    splash_path = _resource_path("Images", "Splash_Screen.png")
    if splash_path.exists():
        pixmap = QPixmap(str(splash_path))
        if not pixmap.isNull():
            screen = app.primaryScreen()
            screen_size = screen.size()
            target_w = screen_size.width() // 2.5
            target_h = screen_size.height() // 2.5
            pixmap = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            splash = QSplashScreen(pixmap)
            splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            splash.show()
            app.processEvents()

    pg.setConfigOptions(antialias=True)

    state = AppState()
    window = MainWindow(state)
    if splash is not None:
        def _show_main_window() -> None:
            window.show()
            splash.finish(window)

        QTimer.singleShot(2000, _show_main_window)
    else:
        window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
