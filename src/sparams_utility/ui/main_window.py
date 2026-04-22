from __future__ import annotations

from importlib import resources
from io import BytesIO
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSettings, Qt, QUrl
from PySide6.QtGui import QCloseEvent, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStyle,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import sparams_utility as pkg
from sparams_utility.circuit_solver import ChannelSimResult
from sparams_utility.models.circuit import DriverSpec
from sparams_utility.models.state import AppState, LoadedTouchstone
from sparams_utility.touchstone_parser import parse_touchstone_string
from sparams_utility.ui.circuit_window import CircuitWindow
from sparams_utility.ui.eye_diagram_window import EyeDiagramWindow
from sparams_utility.ui.plot_window import PlotWindow
from sparams_utility.ui.tdr_window import TdrWindow
from sparams_utility.ui.table_models import MagnitudeTableModel, RawDataTableModel
from sparams_utility.ui.table_window import TableWindow


_RECENT_ENTRIES_MAX = 12


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
        self._project_path: str | None = None
        self._recent_projects: list[str] = []
        self._recent_sparams: list[str] = []
        self._project_data_dir: Path | None = None
        self._window_registry: dict[str, dict[str, dict]] = {
            "sp": {},
            "tdr": {},
            "circuit": {},
        }
        self._open_windows: dict[tuple[str, str], object] = {}
        self._load_recent_entries()

        # MDI area as central widget — all sub-windows live here
        self._mdi = QMdiArea()
        self._mdi.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._mdi.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._mdi.subWindowActivated.connect(self._on_subwindow_activated)

        self._project_tree = QTreeWidget()
        self._project_tree.setHeaderHidden(True)
        self._project_tree.setMinimumWidth(220)
        self._project_tree.itemClicked.connect(self._on_project_tree_item_clicked)

        tree_panel = QWidget()
        tree_layout = QVBoxLayout(tree_panel)
        tree_layout.setContentsMargins(6, 6, 6, 6)
        tree_layout.setSpacing(4)
        tree_layout.addWidget(QLabel("Project"))
        tree_layout.addWidget(self._project_tree)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(tree_panel)
        main_splitter.addWidget(self._mdi)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setSizes([260, 1140])
        self._main_splitter = main_splitter
        self.setCentralWidget(main_splitter)

        # ── File ──────────────────────────────────────────────────────────
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Open Project", self._open_project)
        file_menu.addAction("Save Project", self._save_project)
        file_menu.addAction("Save Project As", self._save_project_as)
        file_menu.addAction("Open File", self._open_files)
        self._recent_projects_menu = file_menu.addMenu("Recent Projects")
        self._recent_sparams_menu = file_menu.addMenu("Recent S-Parameters")
        self._rebuild_recent_file_menus()
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # ── Tables ────────────────────────────────────────────────────────
        self._tables_menu = self.menuBar().addMenu("Tables")

        # ── Charts ────────────────────────────────────────────────────────
        charts_menu = self.menuBar().addMenu("Charts")
        charts_menu.addAction("Open New SP Plot Window", self._open_plot_window)
        charts_menu.addAction("Open new TDR Window", self._open_tdr_window)

        report_menu = self.menuBar().addMenu("Report")
        report_menu.addAction("Create SP Plots Reports", self._create_sp_plots_report)
        report_menu.addAction("Create TDR plots reports", self._create_tdr_plots_report)
        report_menu.addAction("Create EYE diagrams report", self._create_eye_diagrams_report)

        circuit_menu = self.menuBar().addMenu("Circuit")
        circuit_menu.addAction("Open circuit composer", self._open_circuit_window)

        self._active_circuit_menu = self.menuBar().addMenu("Circuit Window")
        self._active_circuit_menu.addAction("Duplicate This circuit...", self._duplicate_active_circuit)
        self._active_circuit_menu.menuAction().setVisible(False)

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
        self._state.files_changed.connect(self._refresh_project_tree)
        self._rebuild_tables_menu()
        self._on_subwindow_activated(None)
        self._refresh_project_tree()

    def _project_tree_project_label(self) -> str:
        if self._project_path:
            return self._tree_display_name(Path(self._project_path).name)
        return "Untitled Project"

    def _tree_display_name(self, name: str) -> str:
        raw_name = str(name).strip()
        if not raw_name:
            return raw_name
        suffix = Path(raw_name).suffix
        if suffix:
            return Path(raw_name).stem or raw_name
        return raw_name

    def _icon_for_kind(self, kind: str):
        style = self.style()
        if kind == "tdr":
            return style.standardIcon(QStyle.SP_DriveNetIcon)
        if kind == "circuit":
            # Try to load the custom circuit icon from resources
            try:
                import importlib.resources as pkg_resources
                from PySide6.QtGui import QIcon, QPixmap
                import sparams_utility.resources
                with pkg_resources.files(sparams_utility.resources).joinpath("circuit_icon.png").open("rb") as f:
                    data = f.read()
                    pixmap = QPixmap()
                    pixmap.loadFromData(data)
                    if not pixmap.isNull():
                        return QIcon(pixmap)
            except Exception:
                pass
            # Fallback to default directory icon
            return style.standardIcon(QStyle.SP_DirIcon)
        if kind == "eye-file":
            # Try to load the custom eye icon from resources
            try:
                import importlib.resources as pkg_resources
                from PySide6.QtGui import QIcon, QPixmap
                import sparams_utility.resources
                with pkg_resources.files(sparams_utility.resources).joinpath("eye_icon.png").open("rb") as f:
                    data = f.read()
                    pixmap = QPixmap()
                    pixmap.loadFromData(data)
                    if not pixmap.isNull():
                        return QIcon(pixmap)
            except Exception:
                pass
            # Fallback to default detailed view icon
            return style.standardIcon(QStyle.SP_FileDialogDetailedView)
        if kind == "sp":
            # Try to load the custom SP plot-window icon from resources
            try:
                import importlib.resources as pkg_resources
                from PySide6.QtGui import QIcon, QPixmap
                import sparams_utility.resources
                with pkg_resources.files(sparams_utility.resources).joinpath("SP_Plot.png").open("rb") as f:
                    data = f.read()
                    pixmap = QPixmap()
                    pixmap.loadFromData(data)
                    if not pixmap.isNull():
                        return QIcon(pixmap)
            except Exception:
                pass
            return style.standardIcon(QStyle.SP_ComputerIcon)
        if kind == "sparam-file":
            # Try to load the custom s-parameter icon from resources
            try:
                import importlib.resources as pkg_resources
                from PySide6.QtGui import QIcon, QPixmap
                import sparams_utility.resources
                with pkg_resources.files(sparams_utility.resources).joinpath("sparam_icon.png").open("rb") as f:
                    data = f.read()
                    pixmap = QPixmap()
                    pixmap.loadFromData(data)
                    if not pixmap.isNull():
                        return QIcon(pixmap)
            except Exception:
                pass
            # Fallback to default file link icon
            return style.standardIcon(QStyle.SP_FileLinkIcon)
        return style.standardIcon(QStyle.SP_FileIcon)

    def _new_entry_id(self, kind: str) -> str:
        return f"{kind}:{uuid4().hex[:10]}"

    def _register_window_entry(self, kind: str, widget: object, *, state: dict | None = None) -> str:
        entry_id = self._new_entry_id(kind)
        window_number = int(getattr(widget, "window_number", 0))
        self._window_registry[kind][entry_id] = {
            "id": entry_id,
            "kind": kind,
            "title": str(widget.windowTitle()),
            "state": state if isinstance(state, dict) else self._snapshot_widget_state(widget),
            "window_number": window_number,
            "is_open": False,
            "eye_file": None,
            "sparam_file": None,
            "sparam_plot_file": None,
            "output_kind": None,
        }
        return entry_id

    def _snapshot_widget_state(self, widget: object) -> dict:
        if isinstance(widget, PlotWindow):
            return widget.export_project_state()
        if isinstance(widget, TdrWindow):
            return widget.export_project_state()
        if isinstance(widget, CircuitWindow):
            return widget.export_project_state()
        return {}

    def _bind_open_window(self, kind: str, entry_id: str, widget: object, sub: QMdiSubWindow) -> None:
        entry = self._window_registry[kind].get(entry_id)
        if entry is None:
            return
        entry["is_open"] = True
        entry["title"] = str(widget.windowTitle())
        entry["window_number"] = int(getattr(widget, "window_number", entry.get("window_number", 0)))
        entry["state"] = self._snapshot_widget_state(widget)
        entry["window_size"] = [sub.width(), sub.height()]
        self._open_windows[(kind, entry_id)] = widget

        if hasattr(widget, "project_modified"):
            widget.project_modified.connect(
                lambda k=kind, eid=entry_id, w=widget: self._sync_window_entry(k, eid, w)
            )
        if isinstance(widget, CircuitWindow):
            widget.eye_result_generated.connect(
                lambda payload, eid=entry_id: self._on_circuit_eye_result_generated(eid, payload)
            )
            widget.sparameter_result_generated.connect(
                lambda payload, eid=entry_id: self._on_circuit_sparameter_result_generated(eid, payload)
            )
        sub.destroyed.connect(lambda *_: self._on_window_closed(kind, entry_id))

    def _ensure_data_dir_for_runtime_exports(self) -> Path | None:
        if self._project_data_dir is not None:
            try:
                self._project_data_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                return None
            return self._project_data_dir

        if self._project_path:
            self._project_data_dir = Path(self._project_path).with_name(Path(self._project_path).stem + "_Data")
        else:
            fallback = f"{pkg.__app_name__.replace(' ', '_')}_Data"
            self._project_data_dir = Path.cwd() / fallback

        try:
            self._project_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return self._project_data_dir

    def _build_eye_binary_payload(
        self,
        result: ChannelSimResult,
        *,
        eye_span_ui: int,
        render_mode: str,
        quality_preset: str,
        stat_enabled: bool,
        noise_rms_mv: float,
        jitter_rms_ps: float,
    ) -> bytes:
        payload = BytesIO()
        np.savez_compressed(
            payload,
            time_s=np.asarray(result.time_s, dtype=float),
            waveform_v=np.asarray(result.waveform_v, dtype=float),
            ui_s=np.asarray([float(result.ui_s)], dtype=float),
            is_differential=np.asarray([1 if result.is_differential else 0], dtype=np.int8),
            driver_spec_json=np.asarray([json.dumps(result.driver_spec.to_dict())]),
            eye_span_ui=np.asarray([int(eye_span_ui)], dtype=np.int32),
            render_mode=np.asarray([str(render_mode)]),
            quality_preset=np.asarray([str(quality_preset)]),
            stat_enabled=np.asarray([1 if stat_enabled else 0], dtype=np.int8),
            noise_rms_mv=np.asarray([float(noise_rms_mv)], dtype=float),
            jitter_rms_ps=np.asarray([float(jitter_rms_ps)], dtype=float),
        )
        return payload.getvalue()

    def _on_circuit_eye_result_generated(self, entry_id: str, payload: object) -> None:
        entry = self._window_registry.get("circuit", {}).get(entry_id)
        if entry is None or not isinstance(payload, dict):
            return

        result = payload.get("result")
        if not isinstance(result, ChannelSimResult):
            return

        data_dir = self._ensure_data_dir_for_runtime_exports()
        if data_dir is None:
            return

        circuit_name = str(payload.get("circuit_name") or entry.get("title") or f"Circuit_{entry_id}")
        eye_file_name = self._sanitize_file_stem(circuit_name, f"Circuit_{entry_id}") + "_Eye.eye"
        eye_path = data_dir / eye_file_name

        try:
            binary_payload = self._build_eye_binary_payload(
                result,
                eye_span_ui=int(payload.get("eye_span_ui", 2)),
                render_mode=str(payload.get("render_mode", "Density")),
                quality_preset=str(payload.get("quality_preset", "Balanced")),
                stat_enabled=bool(payload.get("stat_enabled", False)),
                noise_rms_mv=float(payload.get("noise_rms_mv", 0.0)),
                jitter_rms_ps=float(payload.get("jitter_rms_ps", 0.0)),
            )
            eye_path.write_bytes(binary_payload)
        except Exception:
            return

        old_eye_file = str(entry.get("eye_file") or "").strip()
        old_sparam_file = str(entry.get("sparam_file") or "").strip()
        old_sparam_plot_file = str(entry.get("sparam_plot_file") or "").strip()
        entry["eye_file"] = eye_file_name
        entry["sparam_file"] = None
        entry["sparam_plot_file"] = None
        entry["output_kind"] = "eye"
        if old_eye_file and old_eye_file != eye_file_name:
            old_path = data_dir / old_eye_file
            if old_path.exists():
                try:
                    old_path.unlink()
                except OSError:
                    pass
        for stale_name in (old_sparam_file, old_sparam_plot_file):
            if stale_name:
                stale_path = data_dir / stale_name
                if stale_path.exists():
                    try:
                        stale_path.unlink()
                    except OSError:
                        pass
        self._mark_project_dirty()
        self._refresh_project_tree()

    def _on_circuit_sparameter_result_generated(self, entry_id: str, payload: object) -> None:
        entry = self._window_registry.get("circuit", {}).get(entry_id)
        if entry is None or not isinstance(payload, dict):
            return

        touchstone_text = str(payload.get("touchstone_text") or "")
        nports = int(payload.get("nports") or 0)
        circuit_name = str(payload.get("circuit_name") or entry.get("title") or f"Circuit_{entry_id}")
        if not touchstone_text or nports <= 0:
            return

        data_dir = self._ensure_data_dir_for_runtime_exports()
        if data_dir is None:
            return

        sparam_file_name = f"{self._sanitize_file_stem(circuit_name, f'Circuit_{entry_id}')}.s{nports}p"
        sparam_plot_name = self._sanitize_file_stem(circuit_name, f"Circuit_{entry_id}") + "_S.png"
        sparam_path = data_dir / sparam_file_name
        sparam_plot_path = data_dir / sparam_plot_name

        try:
            sparam_path.write_text(touchstone_text, encoding="utf-8")
            plot_pixmap = self._render_sparameter_plot_pixmap_from_text(
                touchstone_text,
                f"S-Parameters - {circuit_name}",
            )
            if plot_pixmap is None or plot_pixmap.isNull():
                return
            if not plot_pixmap.save(str(sparam_plot_path), "PNG"):
                return
        except Exception:
            return

        old_eye_file = str(entry.get("eye_file") or "").strip()
        old_sparam_file = str(entry.get("sparam_file") or "").strip()
        old_sparam_plot_file = str(entry.get("sparam_plot_file") or "").strip()
        entry["eye_file"] = None
        entry["sparam_file"] = sparam_file_name
        entry["sparam_plot_file"] = sparam_plot_name
        entry["output_kind"] = "sparam"

        if old_eye_file:
            old_eye_path = data_dir / old_eye_file
            if old_eye_path.exists():
                try:
                    old_eye_path.unlink()
                except OSError:
                    pass
        for old_name, new_name in ((old_sparam_file, sparam_file_name), (old_sparam_plot_file, sparam_plot_name)):
            if old_name and old_name != new_name:
                old_path = data_dir / old_name
                if old_path.exists():
                    try:
                        old_path.unlink()
                    except OSError:
                        pass

        self._mark_project_dirty()
        self._refresh_project_tree()

    def _sync_window_entry(self, kind: str, entry_id: str, widget: object) -> None:
        entry = self._window_registry[kind].get(entry_id)
        if entry is None:
            return
        entry["title"] = str(widget.windowTitle())
        entry["state"] = self._snapshot_widget_state(widget)
        entry["is_open"] = True
        parent = widget.parentWidget() if hasattr(widget, "parentWidget") else None
        if isinstance(parent, QMdiSubWindow):
            entry["window_size"] = [parent.width(), parent.height()]
        self._refresh_project_tree()

    def _on_window_closed(self, kind: str, entry_id: str) -> None:
        self._open_windows.pop((kind, entry_id), None)
        entry = self._window_registry[kind].get(entry_id)
        if entry is not None:
            entry["is_open"] = False
        self._refresh_project_tree()

    def _add_tree_window_item(self, parent: QTreeWidgetItem, kind: str, entry_id: str, title: str) -> None:
        entry = self._window_registry[kind].get(entry_id, {})
        label = title.strip()
        if kind == "circuit" and " - " in label:
            label = label.split(" - ", 1)[1].strip() or label
        if kind == "sp" and label.startswith("S-Parameter Plots #"):
            suffix = label.split("#", 1)[1].strip() if "#" in label else ""
            label = f"Plot #{suffix}" if suffix else "Plot"
        if kind == "tdr" and label.startswith("TDR Plots #"):
            suffix = label.split("#", 1)[1].strip() if "#" in label else ""
            label = f"TDR #{suffix}" if suffix else "TDR"

        item = QTreeWidgetItem([label])
        item.setIcon(0, self._icon_for_kind(kind))
        item.setData(0, Qt.UserRole, {"action": "window", "kind": kind, "id": entry_id})
        parent.addChild(item)

        if kind == "circuit":
            eye_file = str(entry.get("eye_file") or "").strip()
            if eye_file:
                eye_item = QTreeWidgetItem([self._tree_display_name(eye_file)])
                eye_item.setIcon(0, self._icon_for_kind("eye-file"))
                eye_item.setData(
                    0,
                    Qt.UserRole,
                    {
                        "action": "output-file",
                        "kind": kind,
                        "id": entry_id,
                        "file": eye_file,
                        "output": "eye",
                    },
                )
                item.addChild(eye_item)
                item.setExpanded(True)

            sparam_plot_file = str(entry.get("sparam_plot_file") or "").strip()
            if sparam_plot_file:
                s_item = QTreeWidgetItem([self._tree_display_name(sparam_plot_file)])
                s_item.setIcon(0, self._icon_for_kind("sparam-file"))
                s_item.setData(
                    0,
                    Qt.UserRole,
                    {
                        "action": "output-file",
                        "kind": kind,
                        "id": entry_id,
                        "file": sparam_plot_file,
                        "output": "sparam-plot",
                    },
                )
                item.addChild(s_item)
                item.setExpanded(True)

    def _refresh_project_tree(self) -> None:
        self._project_tree.clear()

        project_item = QTreeWidgetItem([self._project_tree_project_label()])
        self._project_tree.addTopLevelItem(project_item)

        flat_entries: list[tuple[str, str, str]] = []
        for kind in ("circuit", "sp", "tdr"):
            for entry in self._window_registry[kind].values():
                flat_entries.append((kind, str(entry.get("id", "")), str(entry.get("title", "Window"))))
        flat_entries.sort(key=lambda row: row[2].lower())
        if not flat_entries:
            project_item.addChild(QTreeWidgetItem(["(none)"]))
        else:
            for kind, entry_id, title in flat_entries:
                self._add_tree_window_item(project_item, kind, entry_id, title)

        project_item.setExpanded(True)

    def _on_project_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        payload = item.data(0, Qt.UserRole)
        if not isinstance(payload, dict):
            return
        action = str(payload.get("action", ""))
        if action == "window":
            self._activate_or_open_window_entry(
                str(payload.get("kind", "")),
                str(payload.get("id", "")),
            )
            return
        if action == "output-file":
            self._open_output_file_from_tree(
                str(payload.get("file", "")),
                str(payload.get("output", "")),
            )

    def _activate_or_open_window_entry(self, kind: str, entry_id: str) -> None:
        key = (kind, entry_id)
        open_widget = self._open_windows.get(key)
        if open_widget is not None:
            for sub in self._mdi.subWindowList():
                if sub.widget() is open_widget:
                    self._mdi.setActiveSubWindow(sub)
                    return

        entry = self._window_registry.get(kind, {}).get(entry_id)
        if entry is None:
            return
        state = entry.get("state") if isinstance(entry.get("state"), dict) else {}
        win_number = int(entry.get("window_number", 0))
        window_size = entry.get("window_size")

        if kind == "sp":
            self._plot_counter = max(self._plot_counter, win_number)
            widget = PlotWindow(self._state, window_number=max(1, win_number))
            if state:
                widget.apply_project_state(state)
            widget.project_modified.connect(self._mark_project_dirty)
            sub = self._mdi.addSubWindow(widget)
            if isinstance(window_size, list) and len(window_size) == 2:
                sub.resize(int(window_size[0]), int(window_size[1]))
            else:
                sub.resize(1200, 720)
            widget.show()
            self._bind_open_window("sp", entry_id, widget, sub)
            self._mdi.setActiveSubWindow(sub)
            self._refresh_project_tree()
            return

        if kind == "tdr":
            self._tdr_counter = max(self._tdr_counter, win_number)
            widget = TdrWindow(self._state, window_number=max(1, win_number))
            if state:
                widget.apply_project_state(state)
            widget.project_modified.connect(self._mark_project_dirty)
            sub = self._mdi.addSubWindow(widget)
            if isinstance(window_size, list) and len(window_size) == 2:
                sub.resize(int(window_size[0]), int(window_size[1]))
            else:
                sub.resize(1200, 720)
            widget.show()
            self._bind_open_window("tdr", entry_id, widget, sub)
            self._mdi.setActiveSubWindow(sub)
            self._refresh_project_tree()
            return

        if kind == "circuit":
            self._circuit_counter = max(self._circuit_counter, win_number)
            widget = CircuitWindow(self._state, window_number=max(1, win_number))
            if state:
                widget.apply_project_state(state)
            widget.project_modified.connect(self._mark_project_dirty)
            sub = self._mdi.addSubWindow(widget)
            if isinstance(window_size, list) and len(window_size) == 2:
                sub.resize(int(window_size[0]), int(window_size[1]))
            else:
                sub.resize(1280, 760)
            widget.show()
            self._bind_open_window("circuit", entry_id, widget, sub)
            self._mdi.setActiveSubWindow(sub)
            self._refresh_project_tree()

    def _open_output_file_from_tree(self, file_name: str, output_kind: str) -> None:
        if not file_name:
            return
        if self._project_data_dir is None:
            QMessageBox.information(self, "Output file", "Project data folder is not available.")
            return
        output_path = self._project_data_dir / file_name
        if not output_path.exists():
            QMessageBox.warning(self, "Output file", f"Output file not found:\n{output_path}")
            return

        if output_kind == "eye":
            self._open_eye_binary_file(output_path)
            return

        if output_kind == "sparam-plot":
            pixmap = QPixmap(str(output_path))
            if pixmap.isNull():
                QMessageBox.warning(self, "Output file", f"Could not load plot image:\n{output_path}")
                return
            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            scroll = QScrollArea()
            scroll.setWidget(label)
            scroll.setWidgetResizable(True)
            sub = self._mdi.addSubWindow(scroll)
            sub.setWindowTitle(file_name)
            sub.resize(980, 620)
            scroll.show()
            self._mdi.setActiveSubWindow(sub)

    def _open_eye_binary_file(self, file_path: Path) -> None:
        try:
            with np.load(file_path, allow_pickle=False) as data:
                time_s = np.array(data["time_s"], dtype=float)
                waveform_v = np.array(data["waveform_v"], dtype=float)
                ui_s = float(np.asarray(data["ui_s"]).reshape(-1)[0])
                is_diff = bool(int(np.asarray(data["is_differential"]).reshape(-1)[0]))
                spec_json = str(np.asarray(data["driver_spec_json"]).reshape(-1)[0])
                spec_payload = json.loads(spec_json)
                eye_span_ui = int(np.asarray(data["eye_span_ui"]).reshape(-1)[0])
                render_mode = str(np.asarray(data["render_mode"]).reshape(-1)[0])
                quality_preset = str(np.asarray(data["quality_preset"]).reshape(-1)[0])
                stat_enabled = bool(int(np.asarray(data["stat_enabled"]).reshape(-1)[0]))
                noise_rms_mv = float(np.asarray(data["noise_rms_mv"]).reshape(-1)[0])
                jitter_rms_ps = float(np.asarray(data["jitter_rms_ps"]).reshape(-1)[0])
        except Exception as exc:
            QMessageBox.warning(self, "Eye file", f"Could not load eye binary:\n{exc}")
            return

        result = ChannelSimResult(
            time_s=time_s,
            waveform_v=waveform_v,
            ui_s=ui_s,
            driver_spec=DriverSpec.from_dict(spec_payload),
            is_differential=is_diff,
        )
        eye_win = EyeDiagramWindow(
            result,
            title=file_path.stem,
            parent=self,
            initial_span_ui=eye_span_ui,
            initial_render_mode=render_mode,
            initial_quality_preset=quality_preset,
            statistical_enabled=stat_enabled,
            noise_rms_mv=noise_rms_mv,
            jitter_rms_ps=jitter_rms_ps,
        )
        eye_win.show()

    # ── File loading ──────────────────────────────────────────────────────

    def _settings(self) -> QSettings:
        return QSettings("SPUtility", "SPUtility")

    def _load_recent_entries(self) -> None:
        settings = self._settings()
        recent_projects = settings.value("recent_projects", [], type=list)
        recent_sparams = settings.value("recent_sparams", [], type=list)

        self._recent_projects = [
            str(p) for p in recent_projects if isinstance(p, str) and p.strip()
        ][:_RECENT_ENTRIES_MAX]
        self._recent_sparams = [
            str(p) for p in recent_sparams if isinstance(p, str) and p.strip()
        ][:_RECENT_ENTRIES_MAX]

    def _save_recent_entries(self) -> None:
        settings = self._settings()
        settings.setValue("recent_projects", self._recent_projects)
        settings.setValue("recent_sparams", self._recent_sparams)

    def _normalize_path(self, file_path: str) -> str:
        try:
            return str(Path(file_path).expanduser().resolve())
        except (OSError, RuntimeError):
            return str(file_path)

    def _push_recent_project(self, file_path: str) -> None:
        normalized = self._normalize_path(file_path)
        self._recent_projects = [p for p in self._recent_projects if p != normalized]
        self._recent_projects.insert(0, normalized)
        self._recent_projects = self._recent_projects[:_RECENT_ENTRIES_MAX]
        self._save_recent_entries()
        self._rebuild_recent_file_menus()

    def _push_recent_sparam(self, file_path: str) -> None:
        normalized = self._normalize_path(file_path)
        self._recent_sparams = [p for p in self._recent_sparams if p != normalized]
        self._recent_sparams.insert(0, normalized)
        self._recent_sparams = self._recent_sparams[:_RECENT_ENTRIES_MAX]
        self._save_recent_entries()
        self._rebuild_recent_file_menus()

    def _clear_recent_projects(self) -> None:
        self._recent_projects = []
        self._save_recent_entries()
        self._rebuild_recent_file_menus()

    def _clear_recent_sparams(self) -> None:
        self._recent_sparams = []
        self._save_recent_entries()
        self._rebuild_recent_file_menus()

    def _rebuild_recent_file_menus(self) -> None:
        self._recent_projects_menu.clear()
        if not self._recent_projects:
            action = self._recent_projects_menu.addAction("No recent projects")
            action.setEnabled(False)
        else:
            for index, path in enumerate(self._recent_projects, start=1):
                label = f"{index}. {Path(path).name}"
                action = self._recent_projects_menu.addAction(label)
                action.setToolTip(path)
                action.triggered.connect(
                    lambda checked=False, p=path: self._open_recent_project(p)
                )
            self._recent_projects_menu.addSeparator()
            self._recent_projects_menu.addAction("Clear project history", self._clear_recent_projects)

        self._recent_sparams_menu.clear()
        if not self._recent_sparams:
            action = self._recent_sparams_menu.addAction("No recent S-parameter files")
            action.setEnabled(False)
        else:
            for index, path in enumerate(self._recent_sparams, start=1):
                label = f"{index}. {Path(path).name}"
                action = self._recent_sparams_menu.addAction(label)
                action.setToolTip(path)
                action.triggered.connect(
                    lambda checked=False, p=path: self._open_recent_sparam(p)
                )
            self._recent_sparams_menu.addSeparator()
            self._recent_sparams_menu.addAction("Clear S-parameter history", self._clear_recent_sparams)

    def _open_recent_project(self, file_path: str) -> None:
        if not Path(file_path).exists():
            self._recent_projects = [p for p in self._recent_projects if p != file_path]
            self._save_recent_entries()
            self._rebuild_recent_file_menus()
            QMessageBox.warning(self, "Missing file", f"Project file not found:\n{file_path}")
            return
        self._load_project_from_path(file_path)

    def _open_recent_sparam(self, file_path: str) -> None:
        if not Path(file_path).exists():
            self._recent_sparams = [p for p in self._recent_sparams if p != file_path]
            self._save_recent_entries()
            self._rebuild_recent_file_menus()
            QMessageBox.warning(self, "Missing file", f"S-parameter file not found:\n{file_path}")
            return
        self._load_touchstone_files([file_path])

    def _load_touchstone_files(self, files: list[str]) -> None:
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

        for path in files:
            if Path(path).exists():
                self._push_recent_sparam(path)

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

        self._load_touchstone_files(files)

    def _build_project_payload(self) -> dict:
        loaded_files = self._state.get_loaded_files()
        files_data = [
            {
                "file_path": str(loaded.path),
                "file_name": loaded.display_name,
            }
            for loaded in loaded_files
        ]

        for (kind, entry_id), widget in list(self._open_windows.items()):
            self._sync_window_entry(kind, entry_id, widget)

        plot_windows: list[dict] = []
        tdr_windows: list[dict] = []
        circuit_windows: list[dict] = []
        registry_payload: dict[str, list[dict]] = {"sp": [], "tdr": [], "circuit": []}

        for kind in ("sp", "tdr", "circuit"):
            for entry in self._window_registry[kind].values():
                out = dict(entry)
                registry_payload[kind].append(out)
                if not bool(entry.get("is_open", False)):
                    continue
                state = dict(entry.get("state", {}))
                window_size = entry.get("window_size")
                if isinstance(window_size, list) and len(window_size) == 2:
                    state["window_size"] = [int(window_size[0]), int(window_size[1])]
                if kind == "sp":
                    plot_windows.append(state)
                elif kind == "tdr":
                    tdr_windows.append(state)
                else:
                    circuit_windows.append(state)

        return {
            "app_name": pkg.__app_name__,
            "app_version": pkg.__version__,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "loaded_files": files_data,
            "plots": plot_windows,
            "tdr_plots": tdr_windows,
            "circuits": circuit_windows,
            "window_registry": registry_payload,
        }

    def _save_project_to_path(self, file_path: str) -> bool:
        self._project_data_dir = Path(file_path).with_name(Path(file_path).stem + "_Data")
        export_errors = self._export_project_data_files(self._project_data_dir)
        payload = self._build_project_payload()

        try:
            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save project:\n{exc}")
            return False

        self._project_path = file_path
        self._project_dirty = False
        self._push_recent_project(file_path)
        self._refresh_project_tree()
        if export_errors:
            QMessageBox.information(
                self,
                "Project saved",
                f"Project saved to:\n{file_path}\n\n"
                f"Data folder: {self._project_data_dir}\n\n"
                "Some exports failed:\n" + "\n".join(export_errors),
            )
        else:
            QMessageBox.information(
                self,
                "Project saved",
                f"Project saved to:\n{file_path}\n\nData folder: {self._project_data_dir}",
            )
        return True

    def _sanitize_file_stem(self, text: str, fallback: str) -> str:
        safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in text).strip("_")
        return safe or fallback

    def _render_sparameter_plot_pixmap_from_text(self, touchstone_text: str, title: str) -> QPixmap | None:
        try:
            parsed = parse_touchstone_string(touchstone_text, source_name=title)
        except Exception:
            return None
        freqs = np.array(parsed.magnitude_table.frequencies_hz, dtype=float)
        if freqs.size < 2:
            return None
        trace_names = list(parsed.trace_names)
        if not trace_names:
            return None

        selected = trace_names[: min(6, len(trace_names))]
        traces: list[np.ndarray] = []
        for name in selected:
            values = np.array(parsed.magnitude_table.traces_db.get(name, []), dtype=float)
            if values.size != freqs.size:
                continue
            traces.append(values)
        if not traces:
            return None

        width, height = 1200, 760
        left, right, top, bottom = 90, 30, 50, 70
        plot_w = width - left - right
        plot_h = height - top - bottom

        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#ffffff"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)

        x0, y0 = left, top
        x1, y1 = left + plot_w, top + plot_h
        painter.setPen(QPen(QColor("#d1d5db"), 1))
        for i in range(6):
            y = int(y0 + i * (plot_h / 5.0))
            painter.drawLine(x0, y, x1, y)
        for i in range(6):
            x = int(x0 + i * (plot_w / 5.0))
            painter.drawLine(x, y0, x, y1)

        all_vals = np.concatenate(traces)
        finite = np.isfinite(all_vals)
        if not np.any(finite):
            painter.end()
            return pixmap
        y_min = float(np.min(all_vals[finite]))
        y_max = float(np.max(all_vals[finite]))
        if y_max <= y_min:
            y_max = y_min + 1.0
        x_min = float(np.min(freqs))
        x_max = float(np.max(freqs))
        if x_max <= x_min:
            x_max = x_min + 1.0

        colors = [
            QColor("#1f77b4"), QColor("#d62728"), QColor("#2ca02c"),
            QColor("#ff7f0e"), QColor("#17becf"), QColor("#9467bd"),
        ]

        def _map_x(v: float) -> float:
            return x0 + ((v - x_min) / (x_max - x_min)) * plot_w

        def _map_y(v: float) -> float:
            return y1 - ((v - y_min) / (y_max - y_min)) * plot_h

        for idx, values in enumerate(traces):
            painter.setPen(QPen(colors[idx % len(colors)], 2))
            prev_x = _map_x(float(freqs[0]))
            prev_y = _map_y(float(values[0]))
            for i in range(1, freqs.size):
                x = _map_x(float(freqs[i]))
                y = _map_y(float(values[i]))
                painter.drawLine(int(prev_x), int(prev_y), int(x), int(y))
                prev_x, prev_y = x, y

        painter.setPen(QPen(QColor("#111827"), 2))
        painter.drawRect(x0, y0, plot_w, plot_h)
        painter.drawText(20, 26, title)
        painter.drawText(20, height - 20, "Frequency [Hz]")
        painter.drawText(width - 180, 26, "Magnitude [dB]")
        painter.end()
        return pixmap

    def _export_project_data_files(self, data_dir: Path) -> list[str]:
        errors: list[str] = []
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return [f"Could not create data folder '{data_dir}': {exc}"]

        for entry_id, entry in self._window_registry["circuit"].items():
            circuit_name = str(entry.get("title", f"Circuit {entry_id}"))
            sim_mode = "S-Parameters"
            state_dict = entry.get("state")
            if isinstance(state_dict, dict):
                sim_mode = str(state_dict.get("simulation_mode", "S-Parameters"))

            if sim_mode == "S-Parameters":
                entry["output_kind"] = "sparam"
                entry["eye_file"] = None
                sparam_file = str(entry.get("sparam_file") or "").strip()
                if sparam_file and not (data_dir / sparam_file).exists():
                    errors.append(f"{circuit_name}: missing referenced SnP file ({sparam_file})")
                    entry["sparam_file"] = None
                sparam_plot_file = str(entry.get("sparam_plot_file") or "").strip()
                if sparam_plot_file and not (data_dir / sparam_plot_file).exists():
                    errors.append(f"{circuit_name}: missing referenced S-plot file ({sparam_plot_file})")
                    entry["sparam_plot_file"] = None
            else:
                entry["output_kind"] = "eye"
                entry["sparam_file"] = None
                entry["sparam_plot_file"] = None
                eye_file = str(entry.get("eye_file") or "").strip()
                if eye_file and not (data_dir / eye_file).exists():
                    errors.append(f"{circuit_name}: missing referenced Eye file ({eye_file})")
                    entry["eye_file"] = None

        self._refresh_project_tree()
        return errors

    def _save_project_as(self) -> bool:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            self._project_path or "SPUtility_project.json",
            "SPUtility Project (*.json);;All files (*)",
        )
        if not file_path:
            return False

        return self._save_project_to_path(file_path)

    def _save_project(self) -> bool:
        if self._project_path is None:
            return self._save_project_as()

        return self._save_project_to_path(self._project_path)

    def _open_project(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "SPUtility Project (*.json);;All files (*)",
        )
        if not file_path:
            return

        self._load_project_from_path(file_path)

    def _load_project_from_path(self, file_path: str) -> None:

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
        self._window_registry = {"sp": {}, "tdr": {}, "circuit": {}}
        self._open_windows = {}

        _, errors = self._state.load_files(file_paths)

        restored_plot = 0
        restored_tdr = 0
        restored_circuit = 0

        registry = payload.get("window_registry")
        if isinstance(registry, dict):
            for kind in ("sp", "tdr", "circuit"):
                raw_entries = registry.get(kind, [])
                if not isinstance(raw_entries, list):
                    continue
                for item in raw_entries:
                    if not isinstance(item, dict):
                        continue
                    entry_id = str(item.get("id") or self._new_entry_id(kind))
                    self._window_registry[kind][entry_id] = {
                        "id": entry_id,
                        "kind": kind,
                        "title": str(item.get("title", "Window")),
                        "state": dict(item.get("state", {})) if isinstance(item.get("state"), dict) else {},
                        "window_number": int(item.get("window_number", 0)),
                        "is_open": bool(item.get("is_open", False)),
                        "window_size": item.get("window_size"),
                        "eye_file": item.get("eye_file"),
                        "sparam_file": item.get("sparam_file"),
                        "sparam_plot_file": item.get("sparam_plot_file"),
                        "output_kind": item.get("output_kind"),
                    }

            for kind in ("sp", "tdr", "circuit"):
                for entry_id, entry in self._window_registry[kind].items():
                    if bool(entry.get("is_open", False)):
                        self._activate_or_open_window_entry(kind, entry_id)

            restored_plot = sum(1 for e in self._window_registry["sp"].values() if bool(e.get("is_open", False)))
            restored_tdr = sum(1 for e in self._window_registry["tdr"].values() if bool(e.get("is_open", False)))
            restored_circuit = sum(1 for e in self._window_registry["circuit"].values() if bool(e.get("is_open", False)))
        else:
            # Backward compatibility with older project files.
            for plot_state in payload.get("plots", []):
                if not isinstance(plot_state, dict):
                    continue
                self._plot_counter += 1
                plot_win = PlotWindow(self._state, window_number=self._plot_counter)
                plot_win.apply_project_state(plot_state)
                entry_id = self._register_window_entry("sp", plot_win)
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
                plot_win.project_modified.connect(self._mark_project_dirty)
                plot_win.show()
                self._bind_open_window("sp", entry_id, plot_win, sub)
                restored_plot += 1

            for tdr_state in payload.get("tdr_plots", []):
                if not isinstance(tdr_state, dict):
                    continue
                self._tdr_counter += 1
                tdr_win = TdrWindow(self._state, window_number=self._tdr_counter)
                tdr_win.apply_project_state(tdr_state)
                entry_id = self._register_window_entry("tdr", tdr_win)
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
                tdr_win.project_modified.connect(self._mark_project_dirty)
                tdr_win.show()
                self._bind_open_window("tdr", entry_id, tdr_win, sub)
                restored_tdr += 1

            for circuit_state in payload.get("circuits", []):
                if not isinstance(circuit_state, dict):
                    continue
                self._circuit_counter += 1
                circuit_win = CircuitWindow(self._state, window_number=self._circuit_counter)
                circuit_win.apply_project_state(circuit_state)
                entry_id = self._register_window_entry("circuit", circuit_win)
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
                circuit_win.project_modified.connect(self._mark_project_dirty)
                circuit_win.show()
                self._bind_open_window("circuit", entry_id, circuit_win, sub)
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
        self._project_path = file_path
        self._project_data_dir = Path(file_path).with_name(Path(file_path).stem + "_Data")
        self._project_dirty = False
        self._push_recent_project(file_path)
        self._refresh_project_tree()

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
            self._refresh_project_tree()

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
        entry_id = self._register_window_entry("sp", plot_win)
        self._bind_open_window("sp", entry_id, plot_win, sub)
        self._mdi.setActiveSubWindow(sub)
        self._mark_project_dirty()
        self._refresh_project_tree()

    def _open_tdr_window(self) -> None:
        self._tdr_counter += 1
        tdr_win = TdrWindow(self._state, window_number=self._tdr_counter)
        tdr_win.project_modified.connect(self._mark_project_dirty)
        sub = self._mdi.addSubWindow(tdr_win)
        sub.resize(1200, 720)
        tdr_win.show()
        entry_id = self._register_window_entry("tdr", tdr_win)
        self._bind_open_window("tdr", entry_id, tdr_win, sub)
        self._mdi.setActiveSubWindow(sub)
        self._mark_project_dirty()
        self._refresh_project_tree()

    def _open_circuit_window(self) -> None:
        circuit_number = self._next_available_circuit_number()
        self._circuit_counter = max(self._circuit_counter, circuit_number)
        circuit_win = CircuitWindow(self._state, window_number=circuit_number)
        circuit_win.project_modified.connect(self._mark_project_dirty)
        sub = self._mdi.addSubWindow(circuit_win)
        sub.resize(1280, 760)
        circuit_win.show()
        entry_id = self._register_window_entry("circuit", circuit_win)
        self._bind_open_window("circuit", entry_id, circuit_win, sub)
        self._mdi.setActiveSubWindow(sub)
        self._mark_project_dirty()
        self._refresh_project_tree()

    def _next_available_circuit_number(self) -> int:
        used_numbers: set[int] = set()
        for sub in self._mdi.subWindowList():
            widget = sub.widget()
            if isinstance(widget, CircuitWindow):
                used_numbers.add(int(getattr(widget, "window_number", 0)))
        for entry in self._window_registry["circuit"].values():
            used_numbers.add(int(entry.get("window_number", 0)))
        number = 1
        while number in used_numbers:
            number += 1
        return number

    def _on_subwindow_activated(self, subwindow: QMdiSubWindow | None) -> None:
        widget = subwindow.widget() if subwindow is not None else None
        self._active_circuit_menu.menuAction().setVisible(isinstance(widget, CircuitWindow))
        self._refresh_project_tree()

    def _duplicate_active_circuit(self) -> None:
        active_subwindow = self._mdi.activeSubWindow()
        if active_subwindow is None or not isinstance(active_subwindow.widget(), CircuitWindow):
            QMessageBox.information(
                self,
                "Duplicate circuit",
                "Select a Circuit Composer window before using this command.",
            )
            return

        source_window = active_subwindow.widget()
        source_state = source_window.export_project_state()
        circuit_number = self._next_available_circuit_number()
        self._circuit_counter = max(self._circuit_counter, circuit_number)

        duplicated_window = CircuitWindow(self._state, window_number=circuit_number)
        duplicated_window.apply_project_state(source_state)
        duplicated_window.project_modified.connect(self._mark_project_dirty)

        sub = self._mdi.addSubWindow(duplicated_window)
        sub.resize(active_subwindow.size())
        duplicated_window.show()
        entry_id = self._register_window_entry("circuit", duplicated_window)
        self._bind_open_window("circuit", entry_id, duplicated_window, sub)
        self._mdi.setActiveSubWindow(sub)
        self._mark_project_dirty()
        self._refresh_project_tree()

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

    # ── Report export helpers ────────────────────────────────────────────

    def _ensure_docx_support(self):
        try:
            from docx import Document
            from docx.shared import Inches
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Report unavailable",
                "python-docx is required to create Word reports. "
                f"Install dependency and retry.\nDetails: {exc}",
            )
            return None, None
        return Document, Inches

    def _choose_report_path(self, title: str, default_name: str) -> str | None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            default_name,
            "Word Document (*.docx);;All files (*)",
        )
        if not file_path:
            return None
        if not file_path.lower().endswith(".docx"):
            file_path = f"{file_path}.docx"
        return file_path

    def _pixmap_to_png_stream(self, pixmap) -> BytesIO | None:
        if pixmap is None or pixmap.isNull():
            return None
        raw = QByteArray()
        buf = QBuffer(raw)
        if not buf.open(QIODevice.WriteOnly):
            return None
        ok = pixmap.save(buf, "PNG")
        buf.close()
        if not ok:
            return None
        stream = BytesIO(bytes(raw))
        stream.seek(0)
        return stream

    def _report_header(self, doc, title: str) -> None:
        doc.add_heading(title, level=1)
        doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _create_sp_plots_report(self) -> None:
        windows = [
            sub.widget()
            for sub in self._mdi.subWindowList()
            if isinstance(sub.widget(), PlotWindow)
        ]
        if not windows:
            QMessageBox.information(self, "Report", "No SP Plot windows are currently open.")
            return

        Document, Inches = self._ensure_docx_support()
        if Document is None:
            return
        output_path = self._choose_report_path(
            "Create SP Plots Reports",
            "SP_Plots_Report.docx",
        )
        if output_path is None:
            return

        doc = Document()
        self._report_header(doc, "SP Plots Report")

        for idx, win in enumerate(windows, start=1):
            if idx > 1:
                doc.add_page_break()
            doc.add_heading(f"Plot Window {idx}: {win.windowTitle()}", level=2)
            state = win.export_project_state()
            plotted_files: list[str] = []
            for file_entry in state.get("files", []):
                selected = file_entry.get("selected_parameters", [])
                if not isinstance(selected, list) or len(selected) == 0:
                    continue
                file_path = str(file_entry.get("file_path", "")).strip()
                if not file_path:
                    continue
                plotted_files.append(Path(file_path).name)
            plotted_files = list(dict.fromkeys(plotted_files))
            doc.add_paragraph(
                "Files: " + (", ".join(plotted_files) if plotted_files else "n/a")
            )

            stream = self._pixmap_to_png_stream(win._plot_widget.grab())
            if stream is not None:
                doc.add_picture(stream, width=Inches(6.8))

        try:
            doc.save(output_path)
        except Exception as exc:
            QMessageBox.warning(self, "Report", f"Could not save report:\n{exc}")
            return
        QMessageBox.information(self, "Report", f"SP plots report created:\n{output_path}")

    def _create_tdr_plots_report(self) -> None:
        windows = [
            sub.widget()
            for sub in self._mdi.subWindowList()
            if isinstance(sub.widget(), TdrWindow)
        ]
        if not windows:
            QMessageBox.information(self, "Report", "No TDR Plot windows are currently open.")
            return

        Document, Inches = self._ensure_docx_support()
        if Document is None:
            return
        output_path = self._choose_report_path(
            "Create TDR plots reports",
            "TDR_Plots_Report.docx",
        )
        if output_path is None:
            return

        doc = Document()
        self._report_header(doc, "TDR Plots Report")

        for idx, win in enumerate(windows, start=1):
            if idx > 1:
                doc.add_page_break()
            doc.add_heading(f"TDR Window {idx}: {win.windowTitle()}", level=2)
            state = win.export_project_state()
            plotted_files: list[str] = []
            for file_entry in state.get("files", []):
                if not bool(file_entry.get("show", True)):
                    continue
                selected_trace = str(file_entry.get("selected_trace", "")).strip()
                if not selected_trace:
                    continue
                file_path = str(file_entry.get("file_path", "")).strip()
                if not file_path:
                    continue
                plotted_files.append(Path(file_path).name)
            plotted_files = list(dict.fromkeys(plotted_files))
            doc.add_paragraph(
                "Files: " + (", ".join(plotted_files) if plotted_files else "n/a")
            )

            stream = self._pixmap_to_png_stream(win._plot_widget.grab())
            if stream is not None:
                doc.add_picture(stream, width=Inches(6.8))

        try:
            doc.save(output_path)
        except Exception as exc:
            QMessageBox.warning(self, "Report", f"Could not save report:\n{exc}")
            return
        QMessageBox.information(self, "Report", f"TDR plots report created:\n{output_path}")

    def _create_eye_diagrams_report(self) -> None:
        eye_entries: list[tuple[CircuitWindow, object]] = []
        generation_issues: list[str] = []
        circuit_windows = [
            sub.widget()
            for sub in self._mdi.subWindowList()
            if isinstance(sub.widget(), CircuitWindow)
        ]

        if not circuit_windows:
            QMessageBox.information(self, "Report", "No Circuit windows are currently open.")
            return

        for widget in circuit_windows:
            windows = widget.get_open_eye_windows()
            if not windows:
                generated, err = widget.generate_eye_window_for_report()
                if generated is not None:
                    windows = [generated]
                elif err is not None:
                    generation_issues.append(f"{widget.circuit_display_name()}: {err}")
            for eye_win in windows:
                eye_entries.append((widget, eye_win))

        if not eye_entries:
            details = "\n".join(generation_issues)
            msg = "No Eye diagrams could be generated for the current project windows."
            if details:
                msg += "\n\n" + details
            QMessageBox.information(self, "Report", msg)
            return

        Document, Inches = self._ensure_docx_support()
        if Document is None:
            return
        output_path = self._choose_report_path(
            "Create EYE diagrams report",
            "EYE_Diagrams_Report.docx",
        )
        if output_path is None:
            return

        doc = Document()
        self._report_header(doc, "Eye Diagrams Report")

        def _fmt_mv(v: float) -> str:
            import math as _math
            return "n/a" if not _math.isfinite(v) else f"{v * 1000:.2f} mV"

        def _fmt_ps(v: float) -> str:
            import math as _math
            return "n/a" if not _math.isfinite(v) else f"{v:.2f} ps"

        def _fmt_pct(v: float) -> str:
            import math as _math
            return "n/a" if not _math.isfinite(v) else f"{v:.2f} %"

        for idx, (circuit_win, eye_win) in enumerate(eye_entries, start=1):
            if idx > 1:
                doc.add_page_break()

            state = eye_win.export_report_state()
            meas = state.get("measurements", {})
            files = circuit_win.report_touchstone_file_names()
            circuit_name = circuit_win.circuit_display_name()

            doc.add_heading(f"Eye Diagram {idx}: {eye_win.windowTitle()}", level=2)
            doc.add_paragraph("Circuit: " + circuit_name)
            doc.add_paragraph("Files: " + (", ".join(files) if files else "n/a"))

            settings_lines = [
                f"Mode: {state.get('mode', 'n/a')}",
                f"Bitrate: {float(state.get('bitrate_gbps', 0.0)):.3f} Gbps",
                f"Encoding: {state.get('encoding', 'n/a')}",
                f"PRBS: {state.get('prbs_pattern', 'n/a')}",
                f"Bits: {int(state.get('num_bits', 0))}",
                f"Rise/Fall: {float(state.get('rise_time_ps', 0.0)):.2f} ps / {float(state.get('fall_time_ps', 0.0)):.2f} ps",
                f"Vhigh/Vlow: {float(state.get('voltage_high_v', 0.0)):.4f} V / {float(state.get('voltage_low_v', 0.0)):.4f} V",
                f"Eye span: {int(state.get('eye_span_ui', 0))} UI",
                f"Render mode: {state.get('render_mode', 'n/a')}",
                f"Quality preset: {state.get('quality_preset', 'n/a')}",
                f"Statistical: {'ON' if bool(state.get('statistical_enabled', False)) else 'OFF'}",
                f"Noise RMS: {float(state.get('noise_rms_mv', 0.0)):.2f} mV",
                f"Jitter RMS: {float(state.get('jitter_rms_ps', 0.0)):.2f} ps",
            ]
            doc.add_paragraph("Settings:\n" + "\n".join(settings_lines))

            if isinstance(meas, dict):
                m_lines = [
                    f"One Level: {_fmt_mv(float(meas.get('one_level', float('nan'))))}",
                    f"Zero Level: {_fmt_mv(float(meas.get('zero_level', float('nan'))))}",
                    f"Eye Amplitude: {_fmt_mv(float(meas.get('eye_amplitude', float('nan'))))}",
                    f"Eye Height: {_fmt_mv(float(meas.get('eye_height', float('nan'))))}",
                    f"Eye Width: {_fmt_ps(float(meas.get('eye_width_ps', meas.get('width_ps', float('nan')))))}",
                    f"Eye Crossing: {_fmt_pct(float(meas.get('eye_crossing_pct', float('nan'))))}",
                    f"Bit Period: {_fmt_ps(float(meas.get('bit_period_ps', float('nan'))))}",
                ]
            else:
                m_lines = ["Measurements: n/a"]
            doc.add_paragraph("Measurements:\n" + "\n".join(m_lines))

            eye_stream = self._pixmap_to_png_stream(eye_win.grab_eye_plot_pixmap())
            if eye_stream is not None:
                doc.add_paragraph("Eye Diagram Snapshot")
                doc.add_picture(eye_stream, width=Inches(6.8))

            circuit_stream = self._pixmap_to_png_stream(circuit_win.grab_circuit_snapshot_pixmap())
            if circuit_stream is not None:
                doc.add_paragraph("Related Circuit Snapshot")
                doc.add_picture(circuit_stream, width=Inches(6.8))

        try:
            doc.save(output_path)
        except Exception as exc:
            QMessageBox.warning(self, "Report", f"Could not save report:\n{exc}")
            return

        if generation_issues:
            QMessageBox.information(
                self,
                "Report",
                f"Eye diagrams report created:\n{output_path}\n\n"
                "Some circuit windows could not generate an eye diagram:\n"
                + "\n".join(generation_issues),
            )
        else:
            QMessageBox.information(self, "Report", f"Eye diagrams report created:\n{output_path}")

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
            help_resource = resources.files("sparams_utility.resources.help").joinpath(
                "help_en.html"
            )
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
        with resources.as_file(help_resource) as help_path:
            browser.setSource(QUrl.fromLocalFile(str(help_path)))
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
