from __future__ import annotations

import sys
from pathlib import Path


def _resource_path(*parts: str) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parents[2]
    return base.joinpath(*parts)


def run(app=None, splash=None) -> int:
    """Main entry point.

    Parameters
    ----------
    app:
        A pre-created ``QApplication`` instance.  When *None* (e.g. when this
        module is run directly), a new one is created here.
    splash:
        A ``QSplashScreen`` that is already visible.  When *None* the function
        tries to create and show one itself before loading heavy dependencies.
    """
    # PySide6 is already imported by the bootstrap in app.py when the app is
    # launched normally.  Import it here only as a fallback (direct execution).
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtWidgets import QApplication, QSplashScreen

    standalone = app is None

    if standalone:
        # Running this file directly (dev / test) – replicate bootstrap logic.
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "SParamsUtility.App"
                )
            except Exception:
                pass

        app = QApplication(sys.argv)
        app.setApplicationName("S-Parameters Utility")

        icon_path = _resource_path("Images", "Icon.png")
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))

        splash_path = _resource_path("Images", "Splash_Screen.png")
        if splash_path.exists():
            pixmap = QPixmap(str(splash_path))
            if not pixmap.isNull():
                screen = app.primaryScreen()
                if screen is not None:
                    sz = screen.size()
                    target_w = max(320, int(sz.width() * 0.4))
                    target_h = max(180, int(sz.height() * 0.4))
                    pixmap = pixmap.scaled(
                        target_w, target_h,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                splash = QSplashScreen(pixmap)
                splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
                splash.show()
                app.processEvents()

    # ── Heavy imports – happen while splash is already on screen ──
    import pyqtgraph as pg
    from sparams_utility.models.state import AppState
    from sparams_utility.ui.main_window import MainWindow

    pg.setConfigOptions(antialias=True)

    state = AppState()
    window = MainWindow(state)

    window.show()
    if splash is not None:
        # Defer finish() by one event cycle so the main window gets its
        # first paint before the splash disappears.
        QTimer.singleShot(0, lambda: splash.finish(window))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
