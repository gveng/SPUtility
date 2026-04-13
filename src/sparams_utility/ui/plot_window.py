from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QMainWindow,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sparams_utility.models.state import AppState
from sparams_utility.ui.plot_settings_dialog import PlotSettings, PlotSettingsDialog

_AXIS_PEN = pg.mkPen(color="#222222", width=1)
_TABLE_FONT_PT = 8


class PlotWindow(QMainWindow):
    project_modified = Signal()

    _PLOT_COLORS = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#9467bd",
    ]

    def __init__(self, state: AppState, parent=None, window_number: int = 1) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"S-Parameter Plots #{window_number}")
        self.resize(1250, 800)

        self._state = state
        self._selected_traces: Dict[str, Set[str]] = {}
        self._labels: Dict[str, str] = {}          # file_id -> legend label
        self._row_to_fid: List[str] = []            # row index -> file_id
        self._legend_offset = (10.0, 10.0)
        self._settings = PlotSettings()

        settings_menu = self.menuBar().addMenu("Settings")
        settings_action = settings_menu.addAction("Plot settings")
        settings_action.triggered.connect(self._open_settings_dialog)

        # ── Plot widget ───────────────────────────────────────────────────
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self._plot_widget.setLabel("left", "Magnitude", units="dB")
        # Keep symmetric top/bottom margins so the full plot frame remains visible.
        self._plot_widget.getPlotItem().layout.setContentsMargins(8, 8, 8, 8)

        pi = self._plot_widget.getPlotItem()
        for side in ("bottom", "left", "top", "right"):
            ax = pi.getAxis(side)
            ax.setPen(_AXIS_PEN)
            ax.setTextPen(_AXIS_PEN)
        pi.getAxis("top").setStyle(showValues=False)
        pi.getAxis("right").setStyle(showValues=False)
        # showValues order: (left, top, right, bottom)
        # left=True → Magnitude [dB], bottom=True → Frequency [Hz]
        pi.showAxes(True, showValues=(True, False, False, True))

        self._legend = self._plot_widget.addLegend(offset=self._legend_offset)

        # ── Selection table ───────────────────────────────────────────────
        # Col 0 = File (read-only) | Col 1 = Label (editable) | Col 2..N = traces
        self._selection_table = QTableWidget(0, 2)
        self._selection_table.setHorizontalHeaderLabels(["File", "Legend label"])
        self._selection_table.verticalHeader().setVisible(False)
        self._selection_table.setAlternatingRowColors(True)
        self._selection_table.setSelectionMode(QTableWidget.NoSelection)
        self._selection_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._selection_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Interactive
        )
        self._selection_table.setColumnWidth(1, 200)

        tbl_font = QFont()
        tbl_font.setPointSize(_TABLE_FONT_PT)
        self._selection_table.setFont(tbl_font)
        self._selection_table.horizontalHeader().setFont(tbl_font)
        row_h = tbl_font.pointSize() * 3
        self._selection_table.verticalHeader().setDefaultSectionSize(max(row_h, 22))

        self._selection_table.cellChanged.connect(self._on_cell_changed)

        # ── Splitter: table top (30%), plot bottom (70%) ──────────────────
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._selection_table)
        splitter.addWidget(self._plot_widget)
        splitter.setSizes([240, 560])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

        self._state.files_changed.connect(self.refresh_from_state)
        self.refresh_from_state()

    # ── Table population ──────────────────────────────────────────────────

    def refresh_from_state(self) -> None:
        loaded_files = self._state.get_loaded_files()
        valid_ids = {item.file_id for item in loaded_files}
        self._selected_traces = {
            fid: t for fid, t in self._selected_traces.items() if fid in valid_ids
        }

        # Collect unique trace names in insertion order
        all_traces: List[str] = []
        seen: Set[str] = set()
        for loaded in loaded_files:
            for trace in loaded.data.trace_names:
                if trace not in seen:
                    all_traces.append(trace)
                    seen.add(trace)

        n_trace_cols = len(all_traces)
        n_cols = 2 + n_trace_cols          # File | Label | traces…
        headers = ["File", "Legend label"] + all_traces

        self._selection_table.cellChanged.disconnect(self._on_cell_changed)
        self._selection_table.blockSignals(True)

        self._selection_table.setRowCount(0)
        self._selection_table.setColumnCount(n_cols)
        self._selection_table.setHorizontalHeaderLabels(headers)
        self._selection_table.setRowCount(len(loaded_files))

        self._row_to_fid = []
        for row, loaded in enumerate(loaded_files):
            self._row_to_fid.append(loaded.file_id)

            # Col 0: file name (not editable)
            file_item = QTableWidgetItem(loaded.display_name)
            file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
            self._selection_table.setItem(row, 0, file_item)

            # Col 1: legend label (editable, default = file name)
            label_text = self._labels.get(loaded.file_id, loaded.display_name)
            self._labels.setdefault(loaded.file_id, loaded.display_name)
            label_item = QTableWidgetItem(label_text)
            self._selection_table.setItem(row, 1, label_item)

            # Col 2..N: checkboxes for each trace
            chosen = self._selected_traces.setdefault(loaded.file_id, set())
            file_traces = set(loaded.data.trace_names)

            for col_idx, trace in enumerate(all_traces, start=2):
                if trace in file_traces:
                    cb = QCheckBox()
                    cb.setChecked(trace in chosen)

                    cell = QWidget()
                    cell_layout = QHBoxLayout(cell)
                    cell_layout.addWidget(cb)
                    cell_layout.setAlignment(cb, Qt.AlignCenter)
                    cell_layout.setContentsMargins(0, 0, 0, 0)

                    cb.toggled.connect(
                        lambda checked, fid=loaded.file_id, t=trace: self._on_checkbox_changed(
                            fid, t, checked
                        )
                    )
                    self._selection_table.setCellWidget(row, col_idx, cell)

        self._selection_table.resizeColumnToContents(0)
        header = self._selection_table.horizontalHeader()
        # Set compact fixed width for all trace checkbox columns
        for col_idx in range(2, 2 + len(all_traces)):
            header.setSectionResizeMode(col_idx, QHeaderView.Fixed)
            self._selection_table.setColumnWidth(col_idx, 38)
        self._selection_table.blockSignals(False)
        self._selection_table.cellChanged.connect(self._on_cell_changed)
        self._refresh_plot()

    # ── Signal handlers ───────────────────────────────────────────────────

    def _on_cell_changed(self, row: int, col: int) -> None:
        if col != 1:
            return
        if row >= len(self._row_to_fid):
            return
        fid = self._row_to_fid[row]
        item = self._selection_table.item(row, 1)
        if item is not None:
            self._labels[fid] = item.text()
            self._refresh_plot()
            self.project_modified.emit()

    def _on_checkbox_changed(self, file_id: str, trace: str, checked: bool) -> None:
        chosen = self._selected_traces.setdefault(file_id, set())
        if checked:
            chosen.add(trace)
        else:
            chosen.discard(trace)
        self._refresh_plot()
        self.project_modified.emit()

    def _open_settings_dialog(self) -> None:
        dialog = PlotSettingsDialog(self._settings, self)
        if dialog.exec():
            self._settings = dialog.settings
            self._refresh_plot()
            self.project_modified.emit()

    # ── Plot rendering ────────────────────────────────────────────────────

    def _refresh_plot(self) -> None:
        self._legend_offset = self._get_legend_offset()
        plot_item = self._plot_widget.getPlotItem()
        plot_item.clear()
        self._legend.clear()
        plot_item.setLogMode(self._settings.x_log, self._settings.y_log)

        color_index = 0
        for loaded in self._state.get_loaded_files():
            selected = sorted(self._selected_traces.get(loaded.file_id, set()))
            if not selected:
                continue

            legend_label = self._labels.get(loaded.file_id, loaded.display_name)
            frequencies = np.array(loaded.data.magnitude_table.frequencies_hz, dtype=float)

            for trace in selected:
                values = np.array(loaded.data.magnitude_table.traces_db[trace], dtype=float)
                x_data, y_data = frequencies.copy(), values.copy()

                if self._settings.x_log:
                    mask = x_data > 0
                    x_data, y_data = x_data[mask], y_data[mask]
                if self._settings.y_log:
                    mask = y_data > 0
                    x_data, y_data = x_data[mask], y_data[mask]
                else:
                    finite = np.isfinite(y_data)
                    x_data, y_data = x_data[finite], y_data[finite]

                if len(x_data) == 0:
                    continue

                color = self._PLOT_COLORS[color_index % len(self._PLOT_COLORS)]
                color_index += 1
                curve_label = f"{legend_label} - {trace}"
                plot_item.plot(
                    x_data,
                    y_data,
                    name=curve_label,
                    pen=pg.mkPen(color=color, width=2),
                )

            self._set_legend_offset(self._legend_offset)

        self._apply_ranges()

    def _apply_ranges(self) -> None:
        vb = self._plot_widget.getPlotItem().vb

        if self._settings.x_autorange and self._settings.y_autorange:
            self._plot_widget.autoRange()
            return

        if self._settings.x_autorange:
            vb.enableAutoRange(axis=vb.XAxis)
        elif self._settings.x_min is not None and self._settings.x_max is not None:
            vb.setXRange(self._settings.x_min, self._settings.x_max, padding=0.0)

        if self._settings.y_autorange:
            vb.enableAutoRange(axis=vb.YAxis)
        elif self._settings.y_min is not None and self._settings.y_max is not None:
            vb.setYRange(self._settings.y_min, self._settings.y_max, padding=0.0)

    def apply_project_state(self, state: dict) -> None:
        settings = state.get("plot_settings", {})
        self._settings = PlotSettings(
            x_log=bool(settings.get("x_log", False)),
            y_log=bool(settings.get("y_log", False)),
            x_autorange=bool(settings.get("x_autorange", True)),
            y_autorange=bool(settings.get("y_autorange", True)),
            x_min=settings.get("x_min"),
            x_max=settings.get("x_max"),
            y_min=settings.get("y_min"),
            y_max=settings.get("y_max"),
        )

        loaded_by_path = {
            str(item.path.resolve()): item.file_id
            for item in self._state.get_loaded_files()
        }

        restored_labels: Dict[str, str] = {}
        restored_traces: Dict[str, Set[str]] = {}

        for file_entry in state.get("files", []):
            raw_path = file_entry.get("file_path")
            if not raw_path:
                continue
            file_id = loaded_by_path.get(str(Path(raw_path).resolve()))
            if file_id is None:
                continue

            label = file_entry.get("legend_label")
            if isinstance(label, str) and label:
                restored_labels[file_id] = label

            selected = file_entry.get("selected_parameters", [])
            if isinstance(selected, list):
                restored_traces[file_id] = {str(x) for x in selected}

        if restored_labels:
            self._labels.update(restored_labels)
        if restored_traces:
            self._selected_traces.update(restored_traces)

        legend_position = state.get("legend_position")
        if (
            isinstance(legend_position, list)
            and len(legend_position) == 2
            and all(isinstance(v, (int, float)) for v in legend_position)
        ):
            self._legend_offset = (float(legend_position[0]), float(legend_position[1]))

        self.refresh_from_state()

    def export_project_state(self) -> dict:
        self._legend_offset = self._get_legend_offset()
        per_file = []
        for loaded in self._state.get_loaded_files():
            selected = sorted(self._selected_traces.get(loaded.file_id, set()))
            per_file.append(
                {
                    "file_path": str(loaded.path),
                    "file_name": loaded.display_name,
                    "legend_label": self._labels.get(loaded.file_id, loaded.display_name),
                    "selected_parameters": selected,
                }
            )

        return {
            "window_title": self.windowTitle(),
            "plot_settings": {
                "x_log": self._settings.x_log,
                "y_log": self._settings.y_log,
                "x_autorange": self._settings.x_autorange,
                "y_autorange": self._settings.y_autorange,
                "x_min": self._settings.x_min,
                "x_max": self._settings.x_max,
                "y_min": self._settings.y_min,
                "y_max": self._settings.y_max,
            },
            "legend_position": [self._legend_offset[0], self._legend_offset[1]],
            "files": per_file,
        }

    def _get_legend_offset(self) -> tuple[float, float]:
        try:
            offset = self._legend.opts.get("offset", self._legend_offset)
            if isinstance(offset, (tuple, list)) and len(offset) == 2:
                return float(offset[0]), float(offset[1])
            return self._legend_offset
        except Exception:
            return self._legend_offset

    def _set_legend_offset(self, offset: tuple[float, float]) -> None:
        try:
            self._legend.setOffset(offset)
        except Exception:
            pass

