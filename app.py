from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── Bootstrap: show splash immediately with the bare minimum of imports ──
# PySide6 is the only heavy import needed at this point; everything else
# (pyqtgraph, numpy, scipy, MainWindow, …) is deferred until after the
# splash is already on screen.
from PySide6.QtWidgets import QApplication, QSplashScreen  # noqa: E402
from PySide6.QtCore import Qt                               # noqa: E402
from PySide6.QtGui import QIcon, QPixmap                   # noqa: E402


def _boot_resource(*parts: str) -> Path:
    """Locate a resource whether running from source or a frozen bundle."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).joinpath(*parts)
    return ROOT.joinpath(*parts)


def _bootstrap() -> tuple[QApplication, "QSplashScreen | None"]:
    # Must happen before QApplication on Windows.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "SParamsStudio.App"
            )
        except Exception:
            pass

    _app = QApplication(sys.argv)
    _app.setApplicationName("S-Parameters Utility")
    _app.setApplicationName("S-Params Studio")

    icon_path = _boot_resource("Images", "Icon.png")
    if icon_path.exists():
        _app.setWindowIcon(QIcon(str(icon_path)))

    _splash: QSplashScreen | None = None
    splash_path = _boot_resource("Images", "Splash_Screen.png")
    if splash_path.exists():
        pixmap = QPixmap(str(splash_path))
        if not pixmap.isNull():
            screen = _app.primaryScreen()
            if screen is not None:
                sz = screen.size()
                target_w = max(320, int(sz.width() * 0.4))
                target_h = max(180, int(sz.height() * 0.4))
                pixmap = pixmap.scaled(
                    target_w, target_h,
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation,  # faster than Smooth; splash is temporary
                )
            _splash = QSplashScreen(pixmap)
            _splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            _splash.show()
            _app.processEvents()  # first flush — triggers OS paint
            _app.processEvents()  # second flush — ensures it renders before heavy imports

    return _app, _splash


if __name__ == "__main__":
    # 1. Create QApplication and show splash before ANY other import.
    _app, _splash = _bootstrap()

    # 2. Now import everything else (pyqtgraph, numpy, MainWindow, …).
    #    The splash is already visible, so the user sees it during this phase.
    from sparams_utility.main import run  # noqa: E402

    # 3. Hand off pre-built app + splash to run(); it finishes the rest.
    raise SystemExit(run(app=_app, splash=_splash))
