from __future__ import annotations

import math
from typing import Dict, List, Set

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
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


class PlotWindow(QMainWindow):
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

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("S-Parameter Plots")
        self.resize(1250, 800)

        self._state = state
        self._selected_traces: Dict[str, Set[str]] = {}
        self._settings = PlotSettings()

        settings_menu = self.menuBar().addMenu("Settings")
        settings_action = settings_menu.addAction("Plot settings")
        settings_action.triggered.connect(self._open_settings_dialog)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self._plot_widget.setLabel("left", "Magnitude", units="dB")
        self._plot_widget.getAxis("bottom").setPen(pg.mkPen("#222222"))
        self._plot_widget.getAxis("left").setPen(pg.mkPen("#222222"))
        self._plot_widget.getAxis("bottom").setTextPen(pg.mkPen("#222222"))
        self._plot_widget.getAxis("left").setTextPen(pg.mkPen("#222222"))
        # Draw a solid border around the plot area
        self._plot_widget.getPlotItem().getAxis("top").setStyle(showValues=False)
        self._plot_widget.getPlotItem().getAxis("right").setStyle(showValues=False)
        self._plot_widget.getPlotItem().showAxes(True)
        self._legend = self._plot_widget.addLegend(offset=(10, 10))

        self._selection_table = QTableWidget(0, 1)
        self._selection_table.setHorizontalHeaderLabels(["File"])
        self._selection_table.verticalHeader().setVisible(False)
        self._selection_table.setAlternatingRowColors(True)
        self._selection_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._selection_table.setSelectionMode(QTableWidget.NoSelection)
        self._selection_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )

        # Selection table on top, plot on bottom (30% / 70%)
        total_h = 800
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._selection_table)
        splitter.addWidget(self._plot_widget)
        splitter.setSizes([int(total_h * 0.30), int(total_h * 0.70)])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

        self._state.files_changed.connect(self.refresh_from_state)
        self.refresh_from_state()

    def refresh_from_state(self) -> None:
        loaded_files = self._state.get_loaded_files()
        valid_ids = {item.file_id for item in loaded_files}
        self._selected_traces = {
            fid: traces
            for fid, traces in self._selected_traces.items()
            if fid in valid_ids
        }

        # Collect all unique trace names preserving order across all files
        all_traces: List[str] = []
        seen: Set[str] = set()
        for loaded in loaded_files:
            for trace in loaded.data.trace_names:
                if trace not in seen:
                    all_traces.append(trace)
                    seen.add(trace)

        # Col 0 = File name, col 1..N = one column per unique trace
        self._selection_table.setRowCount(0)
        n_cols = 1 + len(all_traces)
        self._selection_table.setColumnCount(n_cols)
        self._selection_table.setHorizontalHeaderLabels(["File"] + all_traces)
        self._selection_table.setRowCount(len(loaded_files))

        for row, loaded in enumerate(loaded_files):
            file_item = QTableWidgetItem(loaded.display_name)
            self._selection_table.setItem(row, 0, file_item)

            chosen = self._selected_traces.setdefault(loaded.file_id, set())
            file_traces = set(loaded.data.trace_names)

            for col_idx, trace in enumerate(all_traces, start=1):
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
        self._refresh_plot()

    def _on_checkbox_changed(self, file_id: str, trace: str, checked: bool) -> None:
        chosen = self._selected_traces.setdefault(file_id, set())
        if checked:
            chosen.add(trace)
        else:
            chosen.discard(trace)
        self._refresh_plot()

    def _open_settings_dialog(self) -> None:
        dialog = PlotSettingsDialog(self._settings, self)
        if dialog.exec():
            self._settings = dialog.settings
            self._refresh_plot()

    def _refresh_plot(self) -> None:
        plot_item = self._plot_widget.getPlotItem()
        plot_item.clear()
        self._legend.clear()
        plot_item.setLogMode(self._settings.x_log, self._settings.y_log)

        color_index = 0
        for loaded in self._state.get_loaded_files():
            selected = sorted(self._selected_traces.get(loaded.file_id, set()))
            if not selected:
                continue

            frequencies = np.array(loaded.data.magnitude_table.frequencies_hz, dtype=float)
            for trace in selected:
                values = np.array(loaded.data.magnitude_table.traces_db[trace], dtype=float)
                x_data, y_data = frequencies, values

                if self._settings.x_log:
                    mask = x_data > 0
                    x_data = x_data[mask]
                    y_data = y_data[mask]
                if self._settings.y_log:
                    mask = y_data > 0
                    x_data = x_data[mask]
                    y_data = y_data[mask]
                else:
                    finite = np.vectorize(math.isfinite)(y_data)
                    x_data = x_data[finite]
                    y_data = y_data[finite]

                if len(x_data) == 0:
                    continue

                color = self._PLOT_COLORS[color_index % len(self._PLOT_COLORS)]
                color_index += 1
                label = f"{loaded.display_name} - {trace}"
                plot_item.plot(
                    x_data,
                    y_data,
                    name=label,
                    pen=pg.mkPen(color=color, width=2),
                )

        self._apply_ranges()

    def _apply_ranges(self) -> None:
        vb = self._plot_widget.getPlotItem().vb

        if self._settings.x_autorange and self._settings.y_autorange:
            self._plot_widget.autoRange()
        else:
            if self._settings.x_autorange:
                vb.enableAutoRange(axis=vb.XAxis)
            elif self._settings.x_min is not None and self._settings.x_max is not None:
                vb.setXRange(self._settings.x_min, self._settings.x_max, padding=0.0)

            if self._settings.y_autorange:
                vb.enableAutoRange(axis=vb.YAxis)
            elif self._settings.y_min is not None and self._settings.y_max is not None:
                vb.setYRange(self._settings.y_min, self._settings.y_max, padding=0.0)
