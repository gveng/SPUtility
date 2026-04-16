from __future__ import annotations

from importlib import resources
import json
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QTextBrowser,
    QVBoxLayout,
)

import sparams_utility as pkg
from sparams_utility.models.state import AppState, LoadedTouchstone
from sparams_utility.ui.circuit_window import CircuitWindow
from sparams_utility.ui.plot_window import PlotWindow
from sparams_utility.ui.tdr_window import TdrWindow
from sparams_utility.ui.table_models import MagnitudeTableModel, RawDataTableModel
from sparams_utility.ui.table_window import TableWindow


class MainWindow(QMainWindow):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.setWindowTitle(f"{pkg.__app_name__}  {pkg.__version__}")
        self.resize(1400, 900)
        app = QApplication.instance()
        if app is not None:
            self.setWindowIcon(app.windowIcon())

        self._state = state
        self._plot_counter: int = 0
        self._tdr_counter: int = 0
        self._circuit_counter: int = 0
        self._project_dirty: bool = False

        # MDI area as central widget — all sub-windows live here
        self._mdi = QMdiArea()
        self._mdi.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._mdi.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setCentralWidget(self._mdi)

        # ── File ──────────────────────────────────────────────────────────
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Open File", self._open_files)
        file_menu.addAction("Open Project", self._open_project)
        file_menu.addAction("Save Project", lambda: self._save_project())
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # ── Tables ────────────────────────────────────────────────────────
        self._tables_menu = self.menuBar().addMenu("Tables")

        # ── Charts ────────────────────────────────────────────────────────
        charts_menu = self.menuBar().addMenu("Charts")
        charts_menu.addAction("Open plot window", self._open_plot_window)

        circuit_menu = self.menuBar().addMenu("Circuit")
        circuit_menu.addAction("Open circuit composer", self._open_circuit_window)

        # ── TDR ───────────────────────────────────────────────────────────
        tdr_menu = self.menuBar().addMenu("TDR")
        tdr_menu.addAction("Open TDR window", self._open_tdr_window)

        # ── View ──────────────────────────────────────────────────────────
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction("Tile windows", self._mdi.tileSubWindows)
        view_menu.addAction("Cascade windows", self._mdi.cascadeSubWindows)
        view_menu.addSeparator()
        view_menu.addAction("Minimize all", self._minimize_all)
        view_menu.addAction("Restore all", self._restore_all)
        view_menu.addAction("Resize all graph with last graph window", self._resize_all_graph_windows)
        view_menu.addSeparator()
        view_menu.addAction("Close all", self._mdi.closeAllSubWindows)

        # ── Help ──────────────────────────────────────────────────────────
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("User Guide", self._show_help)
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
        elif added_count > 0:
            self._mark_project_dirty()

    def _save_project(self) -> bool:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "SPUtility_project.json",
            "SPUtility Project (*.json);;All files (*)",
        )
        if not file_path:
            return False

        loaded_files = self._state.get_loaded_files()
        files_data = [
            {
                "file_path": str(loaded.path),
                "file_name": loaded.display_name,
            }
            for loaded in loaded_files
        ]

        plot_windows = []
        tdr_windows = []
        circuit_windows = []
        for sub in self._mdi.subWindowList():
            widget = sub.widget()
            if isinstance(widget, PlotWindow):
                state = widget.export_project_state()
                state["window_size"] = [sub.width(), sub.height()]
                plot_windows.append(state)
            elif isinstance(widget, TdrWindow):
                state = widget.export_project_state()
                state["window_size"] = [sub.width(), sub.height()]
                tdr_windows.append(state)
            elif isinstance(widget, CircuitWindow):
                state = widget.export_project_state()
                state["window_size"] = [sub.width(), sub.height()]
                circuit_windows.append(state)

        payload = {
            "app_name": pkg.__app_name__,
            "app_version": pkg.__version__,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "loaded_files": files_data,
            "plots": plot_windows,
            "tdr_plots": tdr_windows,
            "circuits": circuit_windows,
        }

        try:
            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save project:\n{exc}")
            return False

        self._project_dirty = False
        QMessageBox.information(self, "Project saved", f"Project saved to:\n{file_path}")
        return True

    def _open_project(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "SPUtility Project (*.json);;All files (*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Open failed", f"Could not open project:\n{exc}")
            return

        raw_files = payload.get("loaded_files", [])
        file_paths: list[str] = []
        for item in raw_files:
            path = item.get("file_path") if isinstance(item, dict) else None
            if isinstance(path, str) and path:
                file_paths.append(path)

        # Reset current workspace state and windows before loading project.
        self._mdi.closeAllSubWindows()
        self._state.clear_files()
        self._plot_counter = 0
        self._tdr_counter = 0
        self._circuit_counter = 0

        _, errors = self._state.load_files(file_paths)

        restored_plot = 0
        restored_tdr = 0
        for plot_state in payload.get("plots", []):
            if not isinstance(plot_state, dict):
                continue
            self._plot_counter += 1
            plot_win = PlotWindow(self._state, window_number=self._plot_counter)
            sub = self._mdi.addSubWindow(plot_win)
            window_size = plot_state.get("window_size")
            if (
                isinstance(window_size, list)
                and len(window_size) == 2
                and all(isinstance(v, int) for v in window_size)
            ):
                sub.resize(window_size[0], window_size[1])
            else:
                sub.resize(1200, 720)
            plot_win.apply_project_state(plot_state)
            plot_win.project_modified.connect(self._mark_project_dirty)
            plot_win.show()
            restored_plot += 1

        for tdr_state in payload.get("tdr_plots", []):
            if not isinstance(tdr_state, dict):
                continue
            self._tdr_counter += 1
            tdr_win = TdrWindow(self._state, window_number=self._tdr_counter)
            sub = self._mdi.addSubWindow(tdr_win)
            window_size = tdr_state.get("window_size")
            if (
                isinstance(window_size, list)
                and len(window_size) == 2
                and all(isinstance(v, int) for v in window_size)
            ):
                sub.resize(window_size[0], window_size[1])
            else:
                sub.resize(1200, 720)
            tdr_win.apply_project_state(tdr_state)
            tdr_win.project_modified.connect(self._mark_project_dirty)
            tdr_win.show()
            restored_tdr += 1

        restored_circuit = 0
        for circuit_state in payload.get("circuits", []):
            if not isinstance(circuit_state, dict):
                continue
            self._circuit_counter += 1
            circuit_win = CircuitWindow(self._state, window_number=self._circuit_counter)
            sub = self._mdi.addSubWindow(circuit_win)
            window_size = circuit_state.get("window_size")
            if (
                isinstance(window_size, list)
                and len(window_size) == 2
                and all(isinstance(v, int) for v in window_size)
            ):
                sub.resize(window_size[0], window_size[1])
            else:
                sub.resize(1280, 760)
            circuit_win.apply_project_state(circuit_state)
            circuit_win.project_modified.connect(self._mark_project_dirty)
            circuit_win.show()
            restored_circuit += 1

        if errors:
            QMessageBox.warning(
                self,
                "Project loaded with warnings",
                "Some files could not be loaded:\n" + "\n".join(errors),
            )
        else:
            QMessageBox.information(
                self,
                "Project loaded",
                f"Loaded {len(file_paths)} file(s), restored {restored_plot} plot window(s), restored {restored_tdr} TDR window(s), and restored {restored_circuit} circuit window(s).",
            )
        self._project_dirty = False

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
            submenu.addSeparator()
            submenu.addAction(
                "Unload file",
                lambda checked=False, file_id=loaded.file_id: self._unload_file(file_id),
            )

    def _unload_file(self, file_id: str) -> None:
        loaded = self._state.get_file(file_id)
        if loaded is None:
            return

        for sub in self._mdi.subWindowList():
            widget = sub.widget()
            if isinstance(widget, CircuitWindow) and widget.references_file(file_id):
                QMessageBox.warning(
                    self,
                    "File in use",
                    "This file is used by at least one circuit window. Remove its block instances before unloading it.",
                )
                return

        for sub in self._mdi.subWindowList():
            widget = sub.widget()
            if isinstance(widget, TableWindow) and widget.file_id == file_id:
                sub.close()

        removed = self._state.unload_file(file_id)
        if removed:
            self._mark_project_dirty()

    def _show_raw_table(self, loaded: LoadedTouchstone) -> None:
        win = TableWindow(
            f"{loaded.display_name} - Raw data",
            RawDataTableModel(loaded.data),
            file_id=loaded.file_id,
        )
        sub = self._mdi.addSubWindow(win)
        sub.resize(960, 520)
        win.show()

    def _show_magnitude_table(self, loaded: LoadedTouchstone) -> None:
        win = TableWindow(
            f"{loaded.display_name} - Magnitude [dB]",
            MagnitudeTableModel(loaded.data),
            file_id=loaded.file_id,
        )
        sub = self._mdi.addSubWindow(win)
        sub.resize(960, 520)
        win.show()

    # ── Plot window ───────────────────────────────────────────────────────

    def _open_plot_window(self) -> None:
        self._plot_counter += 1
        plot_win = PlotWindow(self._state, window_number=self._plot_counter)
        plot_win.project_modified.connect(self._mark_project_dirty)
        sub = self._mdi.addSubWindow(plot_win)
        sub.resize(1200, 720)
        plot_win.show()
        self._mdi.setActiveSubWindow(sub)
        self._mark_project_dirty()

    def _open_tdr_window(self) -> None:
        self._tdr_counter += 1
        tdr_win = TdrWindow(self._state, window_number=self._tdr_counter)
        tdr_win.project_modified.connect(self._mark_project_dirty)
        sub = self._mdi.addSubWindow(tdr_win)
        sub.resize(1200, 720)
        tdr_win.show()
        self._mdi.setActiveSubWindow(sub)
        self._mark_project_dirty()

    def _open_circuit_window(self) -> None:
        self._circuit_counter += 1
        circuit_win = CircuitWindow(self._state, window_number=self._circuit_counter)
        circuit_win.project_modified.connect(self._mark_project_dirty)
        sub = self._mdi.addSubWindow(circuit_win)
        sub.resize(1280, 760)
        circuit_win.show()
        self._mdi.setActiveSubWindow(sub)
        self._mark_project_dirty()

    def _resize_all_graph_windows(self) -> None:
        plot_windows: list[tuple[QMdiSubWindow, PlotWindow]] = []
        for sub in self._mdi.subWindowList():
            widget = sub.widget()
            if isinstance(widget, PlotWindow):
                plot_windows.append((sub, widget))

        if not plot_windows:
            QMessageBox.information(
                self,
                "No graph windows",
                "There are no graph windows to resize.",
            )
            return

        # Use the latest opened plot window as the reference.
        source_sub, source = max(plot_windows, key=lambda pair: pair[1].window_number)
        state = source.get_graph_layout_state()
        source_size = source_sub.size()

        for sub in self._mdi.subWindowList():
            widget = sub.widget()
            if not isinstance(widget, PlotWindow):
                continue
            if widget is source:
                continue

            sub.resize(source_size)
            widget.apply_graph_layout_state(state)

    # ── View helpers ──────────────────────────────────────────────────────

    def _minimize_all(self) -> None:
        for sub in self._mdi.subWindowList():
            sub.showMinimized()

    def _restore_all(self) -> None:
        for sub in self._mdi.subWindowList():
            sub.showNormal()

    # ── Help > About ──────────────────────────────────────────────────────

    def _show_help(self) -> None:
        try:
            help_html = resources.files("sparams_utility.resources.help").joinpath(
                "help_en.html"
            ).read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
            QMessageBox.warning(
                self,
                "Help unavailable",
                f"Could not load the user guide:\n{exc}",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"{pkg.__app_name__} User Guide")
        dlg.resize(820, 640)

        layout = QVBoxLayout(dlg)
        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        browser.setReadOnly(True)
        browser.setHtml(help_html)
        layout.addWidget(browser)

        dlg.exec()

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

    def _mark_project_dirty(self) -> None:
        self._project_dirty = True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._project_dirty:
            event.accept()
            return

        box = QMessageBox(self)
        box.setWindowTitle("Unsaved Changes")
        box.setText("Project has unsaved changes. Do you want to save before closing?")
        box.setIcon(QMessageBox.Warning)
        box.setStandardButtons(
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        box.setDefaultButton(QMessageBox.Save)

        choice = box.exec()
        if choice == QMessageBox.Save:
            if self._save_project():
                event.accept()
            else:
                event.ignore()
            return

        if choice == QMessageBox.Discard:
            event.accept()
            return

        event.ignore()
