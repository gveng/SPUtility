from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QVBoxLayout,
)

import sparams_utility as pkg
from sparams_utility.models.state import AppState, LoadedTouchstone
from sparams_utility.ui.plot_window import PlotWindow
from sparams_utility.ui.table_models import MagnitudeTableModel, RawDataTableModel
from sparams_utility.ui.table_window import TableWindow


class MainWindow(QMainWindow):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.setWindowTitle(f"{pkg.__app_name__}  {pkg.__version__}")
        self.resize(1400, 900)

        self._state = state
        self._plot_subwin: QMdiSubWindow | None = None

        # MDI area as central widget — all sub-windows live here
        self._mdi = QMdiArea()
        self._mdi.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._mdi.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setCentralWidget(self._mdi)

        # ── File ──────────────────────────────────────────────────────────
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Open File", self._open_files)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # ── Tables ────────────────────────────────────────────────────────
        self._tables_menu = self.menuBar().addMenu("Tables")

        # ── Charts ────────────────────────────────────────────────────────
        charts_menu = self.menuBar().addMenu("Charts")
        charts_menu.addAction("Open plot window", self._open_plot_window)

        # ── View ──────────────────────────────────────────────────────────
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction("Tile windows", self._mdi.tileSubWindows)
        view_menu.addAction("Cascade windows", self._mdi.cascadeSubWindows)
        view_menu.addSeparator()
        view_menu.addAction("Minimize all", self._minimize_all)
        view_menu.addAction("Restore all", self._restore_all)
        view_menu.addSeparator()
        view_menu.addAction("Close all", self._mdi.closeAllSubWindows)

        # ── Help ──────────────────────────────────────────────────────────
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About", self._show_about)

        self._state.files_changed.connect(self._rebuild_tables_menu)
        self._rebuild_tables_menu()

    # ── File loading ──────────────────────────────────────────────────────

    def _open_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Touchstone file",
            "",
            "Touchstone (*.s1p *.s2p *.s3p *.s4p *.s5p *.s6p *.s7p *.s8p *.s9p"
            " *.s10p *.s12p *.s16p *.ts);;All files (*)",
        )
        if not files:
            return

        added_count, errors = self._state.load_files(files)

        if errors:
            QMessageBox.warning(self, "Load errors", "\n".join(errors))

        if added_count == 0 and not errors:
            QMessageBox.information(
                self, "No new files", "The selected files are already loaded."
            )

    # ── Tables menu ───────────────────────────────────────────────────────

    def _rebuild_tables_menu(self) -> None:
        self._tables_menu.clear()
        loaded_files = self._state.get_loaded_files()
        if not loaded_files:
            a = self._tables_menu.addAction("No files loaded")
            a.setEnabled(False)
            return

        for loaded in loaded_files:
            submenu = self._tables_menu.addMenu(loaded.display_name)
            submenu.addAction(
                "Raw data table",
                lambda checked=False, item=loaded: self._show_raw_table(item),
            )
            submenu.addAction(
                "Magnitude table [dB]",
                lambda checked=False, item=loaded: self._show_magnitude_table(item),
            )

    def _show_raw_table(self, loaded: LoadedTouchstone) -> None:
        win = TableWindow(
            f"{loaded.display_name} - Raw data", RawDataTableModel(loaded.data)
        )
        sub = self._mdi.addSubWindow(win)
        sub.resize(960, 520)
        win.show()

    def _show_magnitude_table(self, loaded: LoadedTouchstone) -> None:
        win = TableWindow(
            f"{loaded.display_name} - Magnitude [dB]",
            MagnitudeTableModel(loaded.data),
        )
        sub = self._mdi.addSubWindow(win)
        sub.resize(960, 520)
        win.show()

    # ── Plot window ───────────────────────────────────────────────────────

    def _open_plot_window(self) -> None:
        if self._plot_subwin is None or self._plot_subwin not in self._mdi.subWindowList():
            plot_win = PlotWindow(self._state)
            self._plot_subwin = self._mdi.addSubWindow(plot_win)
            self._plot_subwin.resize(1200, 720)
        self._plot_subwin.showNormal()
        self._mdi.setActiveSubWindow(self._plot_subwin)

    # ── View helpers ──────────────────────────────────────────────────────

    def _minimize_all(self) -> None:
        for sub in self._mdi.subWindowList():
            sub.showMinimized()

    def _restore_all(self) -> None:
        for sub in self._mdi.subWindowList():
            sub.showNormal()

    # ── Help > About ──────────────────────────────────────────────────────

    def _show_about(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"About {pkg.__app_name__}")
        dlg.setFixedSize(340, 200)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(8)

        for html in (
            f"<b style='font-size:16px'>{pkg.__app_name__}</b>",
            f"Version {pkg.__version__}",
            "<hr>",
            f"<b>Author:</b> {pkg.__author__}",
            "",
            "S-Parameters Touchstone Viewer",
        ):
            lbl = QLabel(html)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setTextFormat(Qt.RichText)
            layout.addWidget(lbl)

        dlg.exec()
