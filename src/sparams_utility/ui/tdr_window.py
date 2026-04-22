from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from scipy import fft, interpolate, ndimage, signal
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sparams_utility.models.state import AppState

_AXIS_PEN = pg.mkPen(color="#222222", width=1)
_TABLE_FONT_PT = 8


class TdrWindow(QMainWindow):
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
        self.window_number = window_number
        self._tdr_name = f"TDR #{window_number}"
        self._refresh_window_title()
        self.resize(1250, 800)
        app = QApplication.instance()
        if app is not None:
            self.setWindowIcon(app.windowIcon())

        self._state = state
        self._labels: Dict[str, str] = {}
        self._selected_trace: Dict[str, str] = {}
        self._enabled: Dict[str, bool] = {}
        self._row_to_fid: List[str] = []
        self._legend_offset = (10.0, 10.0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._plot_widget.setLabel("bottom", "Time", units="s")
        self._plot_widget.setLabel("left", "Impedance", units="Ohm")
        self._plot_widget.getPlotItem().layout.setContentsMargins(8, 8, 8, 8)

        pi = self._plot_widget.getPlotItem()
        for side in ("bottom", "left", "top", "right"):
            ax = pi.getAxis(side)
            ax.setPen(_AXIS_PEN)
            ax.setTextPen(_AXIS_PEN)
        pi.getAxis("top").setStyle(showValues=False)
        pi.getAxis("right").setStyle(showValues=False)
        pi.showAxes(True, showValues=(True, False, False, True))

        self._legend = self._plot_widget.addLegend(offset=self._legend_offset)

        self._selection_table = QTableWidget(0, 4)
        self._selection_table.setHorizontalHeaderLabels(
            ["File", "Legend label", "Trace", "Show"]
        )
        self._selection_table.verticalHeader().setVisible(False)
        self._selection_table.setAlternatingRowColors(True)
        self._selection_table.setSelectionMode(QTableWidget.NoSelection)
        self._selection_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._selection_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Interactive
        )
        self._selection_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self._selection_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Fixed
        )
        self._selection_table.setColumnWidth(1, 220)
        self._selection_table.setColumnWidth(3, 46)

        tbl_font = QFont()
        tbl_font.setPointSize(_TABLE_FONT_PT)
        self._selection_table.setFont(tbl_font)
        self._selection_table.horizontalHeader().setFont(tbl_font)
        row_h = tbl_font.pointSize() * 3
        self._selection_table.verticalHeader().setDefaultSectionSize(max(row_h, 22))
        self._selection_table.cellChanged.connect(self._on_cell_changed)

        # Top-right controls above TDR plot
        self._rise_time_ps = QDoubleSpinBox()
        self._rise_time_ps.setDecimals(1)
        self._rise_time_ps.setRange(0.0, 1_000_000.0)
        self._rise_time_ps.setValue(0.0)
        self._rise_time_ps.setSuffix(" ps")
        self._rise_time_ps.valueChanged.connect(self._on_tdr_param_changed)

        self._zero_before_front_ns = QDoubleSpinBox()
        self._zero_before_front_ns.setDecimals(3)
        self._zero_before_front_ns.setRange(0.0, 1_000_000.0)
        self._zero_before_front_ns.setValue(0.0)
        self._zero_before_front_ns.setSuffix(" ns")
        self._zero_before_front_ns.valueChanged.connect(self._on_tdr_param_changed)

        self._total_time_ns = QDoubleSpinBox()
        self._total_time_ns.setDecimals(3)
        self._total_time_ns.setRange(0.001, 1_000_000.0)
        self._total_time_ns.setValue(10.0)
        self._total_time_ns.setSuffix(" ns")
        self._total_time_ns.valueChanged.connect(self._on_tdr_param_changed)

        self._window_combo = QComboBox()
        self._window_combo.addItems(["Hann", "Hamming", "Blackman", "Flattop", "None"])
        self._window_combo.setCurrentText("Hann")
        self._window_combo.currentTextChanged.connect(self._on_tdr_param_changed)

        self._tdr_name_edit = QLineEdit(self._tdr_name)
        self._tdr_name_edit.textChanged.connect(self._on_tdr_name_changed)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Plot name:"))
        controls_row.addWidget(self._tdr_name_edit)
        controls_row.addSpacing(8)
        controls_row.addStretch(1)
        controls_row.addWidget(QLabel("Rise time:"))
        controls_row.addWidget(self._rise_time_ps)
        controls_row.addSpacing(10)
        controls_row.addWidget(QLabel("Zero before front:"))
        controls_row.addWidget(self._zero_before_front_ns)
        controls_row.addSpacing(10)
        controls_row.addWidget(QLabel("Total time:"))
        controls_row.addWidget(self._total_time_ns)
        controls_row.addSpacing(10)
        controls_row.addWidget(QLabel("Window:"))
        controls_row.addWidget(self._window_combo)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        plot_layout.addLayout(controls_row)
        plot_layout.addWidget(self._plot_widget)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._selection_table)
        splitter.addWidget(plot_container)
        splitter.setSizes([240, 560])
        self._splitter = splitter

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

        self._state.files_changed.connect(self.refresh_from_state)
        self.refresh_from_state()

    def _refresh_window_title(self) -> None:
        self.setWindowTitle(self._tdr_name.strip() or f"TDR #{self.window_number}")

    def _on_tdr_name_changed(self, value: str) -> None:
        self._tdr_name = value.strip() or f"TDR #{self.window_number}"
        self._refresh_window_title()
        self.project_modified.emit()

    def refresh_from_state(self) -> None:
        loaded_files = self._state.get_loaded_files()
        valid_ids = {item.file_id for item in loaded_files}
        self._labels = {k: v for k, v in self._labels.items() if k in valid_ids}
        self._selected_trace = {
            k: v for k, v in self._selected_trace.items() if k in valid_ids
        }
        self._enabled = {k: v for k, v in self._enabled.items() if k in valid_ids}

        self._selection_table.cellChanged.disconnect(self._on_cell_changed)
        self._selection_table.blockSignals(True)
        self._selection_table.setRowCount(len(loaded_files))

        self._row_to_fid = []
        for row, loaded in enumerate(loaded_files):
            self._row_to_fid.append(loaded.file_id)

            file_item = QTableWidgetItem(loaded.display_name)
            file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
            self._selection_table.setItem(row, 0, file_item)

            label_text = self._labels.get(loaded.file_id, loaded.display_name)
            self._labels.setdefault(loaded.file_id, loaded.display_name)
            label_item = QTableWidgetItem(label_text)
            self._selection_table.setItem(row, 1, label_item)

            traces = [
                t
                for t in loaded.data.trace_names
                if re.fullmatch(r"S(\d+)\1", t)
            ]
            default_trace = "S11" if "S11" in traces else (traces[0] if traces else "")
            selected_trace = self._selected_trace.get(loaded.file_id, default_trace)
            if selected_trace not in traces:
                selected_trace = default_trace
            self._selected_trace[loaded.file_id] = selected_trace

            combo = QComboBox()
            combo.addItems(traces)
            combo.setCurrentText(selected_trace)
            combo.setEnabled(bool(traces))
            combo.currentTextChanged.connect(
                lambda text, fid=loaded.file_id: self._on_trace_changed(fid, text)
            )
            self._selection_table.setCellWidget(row, 2, combo)

            enabled = self._enabled.get(loaded.file_id, True)
            self._enabled.setdefault(loaded.file_id, enabled)
            cb = QCheckBox()
            cb.setChecked(enabled)
            cb.toggled.connect(
                lambda checked, fid=loaded.file_id: self._on_enabled_changed(fid, checked)
            )
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.addWidget(cb)
            cell_layout.setAlignment(cb, Qt.AlignCenter)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            self._selection_table.setCellWidget(row, 3, cell)

        self._selection_table.blockSignals(False)
        self._selection_table.cellChanged.connect(self._on_cell_changed)
        self._refresh_plot()

    def _on_cell_changed(self, row: int, col: int) -> None:
        if col != 1 or row >= len(self._row_to_fid):
            return
        fid = self._row_to_fid[row]
        item = self._selection_table.item(row, 1)
        if item is not None:
            self._labels[fid] = item.text()
            self._refresh_plot()
            self.project_modified.emit()

    def _on_trace_changed(self, file_id: str, trace: str) -> None:
        self._selected_trace[file_id] = trace
        self._refresh_plot()
        self.project_modified.emit()

    def _on_enabled_changed(self, file_id: str, checked: bool) -> None:
        self._enabled[file_id] = checked
        self._refresh_plot()
        self.project_modified.emit()

    def _on_tdr_param_changed(self) -> None:
        self._refresh_plot()
        self.project_modified.emit()

    def _refresh_plot(self) -> None:
        self._legend_offset = self._get_legend_offset()
        plot_item = self._plot_widget.getPlotItem()
        plot_item.clear()
        self._legend.clear()

        color_index = 0
        for loaded in self._state.get_loaded_files():
            if not self._enabled.get(loaded.file_id, True):
                continue

            trace = self._selected_trace.get(loaded.file_id)
            if not trace:
                continue

            t_s, z_ohm = self._compute_tdr_impedance(
                loaded.data.points,
                trace,
                float(loaded.data.options.reference_resistance),
            )
            if t_s is None or z_ohm is None:
                continue

            t_s, z_ohm = self._apply_tdr_plot_params(
                t_s,
                z_ohm,
                float(loaded.data.options.reference_resistance),
            )
            if t_s.size == 0:
                continue

            color = self._PLOT_COLORS[color_index % len(self._PLOT_COLORS)]
            color_index += 1
            legend_label = self._labels.get(loaded.file_id, loaded.display_name)
            curve_label = f"{legend_label} - {trace}"
            plot_item.plot(
                t_s,
                z_ohm,
                name=curve_label,
                pen=pg.mkPen(color=color, width=2),
            )

        self._set_legend_offset(self._legend_offset)

    def apply_project_state(self, state: dict) -> None:
        tdr_name = str(state.get("tdr_name", "")).strip()
        if not tdr_name:
            title = str(state.get("window_title", "")).strip()
            if title:
                tdr_name = title
        if not tdr_name:
            tdr_name = f"TDR #{self.window_number}"
        self._tdr_name = tdr_name
        self._tdr_name_edit.blockSignals(True)
        self._tdr_name_edit.setText(tdr_name)
        self._tdr_name_edit.blockSignals(False)
        self._refresh_window_title()

        loaded_by_path = {
            str(item.path.resolve()): item.file_id
            for item in self._state.get_loaded_files()
        }

        restored_labels: Dict[str, str] = {}
        restored_traces: Dict[str, str] = {}
        restored_enabled: Dict[str, bool] = {}

        for file_entry in state.get("files", []):
            raw_path = file_entry.get("file_path") if isinstance(file_entry, dict) else None
            if not isinstance(raw_path, str) or not raw_path:
                continue

            file_id = loaded_by_path.get(raw_path)
            if file_id is None:
                continue

            label = file_entry.get("legend_label")
            if isinstance(label, str) and label:
                restored_labels[file_id] = label

            selected_trace = file_entry.get("selected_trace")
            if isinstance(selected_trace, str) and selected_trace:
                restored_traces[file_id] = selected_trace

            show = file_entry.get("show")
            if isinstance(show, bool):
                restored_enabled[file_id] = show

        if restored_labels:
            self._labels.update(restored_labels)
        if restored_traces:
            self._selected_trace.update(restored_traces)
        if restored_enabled:
            self._enabled.update(restored_enabled)

        tdr_settings = state.get("tdr_settings", {})
        if isinstance(tdr_settings, dict):
            rise_time_ps = tdr_settings.get("rise_time_ps")
            zero_before_front_ns = tdr_settings.get("zero_before_front_ns")
            total_time_ns = tdr_settings.get("total_time_ns")
            window_name = tdr_settings.get("window")

            if isinstance(rise_time_ps, (int, float)):
                self._rise_time_ps.setValue(float(rise_time_ps))
            if isinstance(zero_before_front_ns, (int, float)):
                self._zero_before_front_ns.setValue(float(zero_before_front_ns))
            if isinstance(total_time_ns, (int, float)):
                self._total_time_ns.setValue(float(total_time_ns))
            if isinstance(window_name, str):
                idx = self._window_combo.findText(window_name)
                if idx >= 0:
                    self._window_combo.setCurrentIndex(idx)

        legend_position = state.get("legend_position")
        if (
            isinstance(legend_position, list)
            and len(legend_position) == 2
            and all(isinstance(v, (int, float)) for v in legend_position)
        ):
            self._legend_offset = (float(legend_position[0]), float(legend_position[1]))

        self.refresh_from_state()

        splitter_sizes = state.get("splitter_sizes")
        if (
            isinstance(splitter_sizes, list)
            and len(splitter_sizes) == 2
            and all(isinstance(v, int) for v in splitter_sizes)
        ):
            self._splitter.setSizes(splitter_sizes)

    def export_project_state(self) -> dict:
        self._legend_offset = self._get_legend_offset()

        per_file = []
        for loaded in self._state.get_loaded_files():
            per_file.append(
                {
                    "file_path": str(loaded.path),
                    "file_name": loaded.display_name,
                    "legend_label": self._labels.get(loaded.file_id, loaded.display_name),
                    "selected_trace": self._selected_trace.get(loaded.file_id, ""),
                    "show": bool(self._enabled.get(loaded.file_id, True)),
                }
            )

        return {
            "window_title": self.windowTitle(),
            "tdr_name": self._tdr_name,
            "tdr_settings": {
                "rise_time_ps": float(self._rise_time_ps.value()),
                "zero_before_front_ns": float(self._zero_before_front_ns.value()),
                "total_time_ns": float(self._total_time_ns.value()),
                "window": self._window_combo.currentText(),
            },
            "legend_position": [self._legend_offset[0], self._legend_offset[1]],
            "splitter_sizes": self._splitter.sizes(),
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

    def _apply_tdr_plot_params(
        self, time_s: np.ndarray, z_ohm: np.ndarray, zref_ohm: float
    ):
        if time_s.size < 2 or z_ohm.size < 2:
            return time_s, z_ohm

        dt = float(time_s[1] - time_s[0])
        if not np.isfinite(dt) or dt <= 0:
            return time_s, z_ohm

        z_out = z_ohm

        # Set the duration of the flat 0-level section before the front.
        zero_before_s = float(self._zero_before_front_ns.value()) * 1e-9
        if zero_before_s > 0.0:
            shift_samples = int(round(zero_before_s / dt))
            if shift_samples > 0:
                z_shifted = np.full_like(z_out, zref_ohm)
                if shift_samples < z_out.size:
                    z_shifted[shift_samples:] = z_out[:-shift_samples]
                z_out = z_shifted

        # Approximate rise-time bandwidth effect with gaussian smoothing.
        rise_ps = float(self._rise_time_ps.value())
        if rise_ps > 0.0:
            rise_s = rise_ps * 1e-12
            sigma_samples = rise_s / (2.355 * dt)
            if sigma_samples >= 0.5:
                if np.any(~np.isfinite(z_out)):
                    valid = np.isfinite(z_out)
                    if np.any(valid):
                        idx = np.arange(z_out.size, dtype=float)
                        z_filled = np.interp(idx, idx[valid], z_out[valid])
                    else:
                        z_filled = z_out
                else:
                    z_filled = z_out

                z_smoothed = ndimage.gaussian_filter1d(
                    z_filled,
                    sigma=sigma_samples,
                    mode="nearest",
                )
                if np.any(~np.isfinite(z_out)):
                    z_smoothed[~np.isfinite(z_out)] = np.nan
                z_out = z_smoothed

        total_s = float(self._total_time_ns.value()) * 1e-9

        t_out = time_s
        t_min = 0.0
        t_max = total_s
        mask = (t_out >= t_min) & (t_out <= t_max)
        return t_out[mask], z_out[mask]

    def _compute_tdr_impedance(self, points, trace_name: str, zref_ohm: float):
        if len(points) < 2:
            return None, None

        match = re.fullmatch(r"S(\d+)(\d+)", trace_name)
        if not match:
            return None, None
        row = int(match.group(1)) - 1
        col = int(match.group(2)) - 1
        if row != col:
            # Impedance conversion is physically meaningful only for reflection traces Sii.
            return None, None

        freqs = []
        gamma = []
        for point in points:
            if row >= len(point.s_matrix) or col >= len(point.s_matrix[row]):
                return None, None
            freqs.append(float(point.frequency_hz))
            gamma.append(point.s_matrix[row][col].complex_value)

        f = np.array(freqs, dtype=float)
        g = np.array(gamma, dtype=complex)

        order = np.argsort(f)
        f = f[order]
        g = g[order]

        f_unique, first_idx = np.unique(f, return_index=True)
        g_unique = g[first_idx]
        if f_unique.size < 2:
            return None, None

        df_samples = np.diff(f_unique)
        df_med = float(np.median(df_samples))
        tol = max(abs(df_med) * 1e-4, 1e-3)
        is_uniform = bool(np.all(np.abs(df_samples - df_med) <= tol))

        if is_uniform:
            f_uniform = f_unique.copy()
            g_uniform = g_unique.copy()
        else:
            # Resample only when input frequency spacing is non-uniform.
            n_pos = max(1024, int(2 ** np.ceil(np.log2(f_unique.size))))
            f_uniform = np.linspace(f_unique[0], f_unique[-1], n_pos)

            interp_real = interpolate.interp1d(
                f_unique,
                g_unique.real,
                kind="linear",
                bounds_error=False,
                fill_value=(g_unique.real[0], g_unique.real[-1]),
                assume_sorted=True,
            )
            interp_imag = interpolate.interp1d(
                f_unique,
                g_unique.imag,
                kind="linear",
                bounds_error=False,
                fill_value=(g_unique.imag[0], g_unique.imag[-1]),
                assume_sorted=True,
            )
            g_real = interp_real(f_uniform)
            g_imag = interp_imag(f_uniform)
            g_uniform = g_real + 1j * g_imag

        if f_uniform[0] > 0.0:
            if f_uniform.size >= 2 and f_uniform[0] <= 0.1 * (f_uniform[1] - f_uniform[0]):
                # First sample is near DC: reuse it as DC to avoid unnecessary extrapolation bias.
                f_uniform[0] = 0.0
                g_uniform[0] = complex(float(np.clip(g_uniform[0].real, -0.999, 0.999)), 0.0)
            else:
                fit_pts = min(6, f_uniform.size)
                if fit_pts >= 2:
                    coeff_r = np.polyfit(
                        f_uniform[:fit_pts],
                        g_uniform.real[:fit_pts],
                        deg=1,
                    )
                    coeff_i = np.polyfit(
                        f_uniform[:fit_pts],
                        g_uniform.imag[:fit_pts],
                        deg=1,
                    )
                    g0_real = float(coeff_r[-1])
                    g0_imag = float(coeff_i[-1])
                else:
                    g0_real = float(g_uniform.real[0])
                    g0_imag = float(g_uniform.imag[0])
                g0_real = float(np.clip(g0_real, -0.999, 0.999))
                g0 = complex(g0_real, g0_imag)
                f_uniform = np.insert(f_uniform, 0, 0.0)
                g_uniform = np.insert(g_uniform, 0, g0)
        else:
            g_uniform[0] = complex(float(np.clip(g_uniform[0].real, -0.999, 0.999)), 0.0)

        if f_uniform.size < 3:
            return None, None

        df = float(f_uniform[1] - f_uniform[0])
        if not np.isfinite(df) or df <= 0:
            return None, None

        window = self._build_frequency_window(g_uniform.size)
        g_windowed = g_uniform * window
        g_windowed[0] = complex(g_uniform[0].real, 0.0)

        n_time = 2 * (g_windowed.size - 1)
        impulse = fft.irfft(g_windowed, n=n_time)
        dt = 1.0 / (n_time * df)
        rho_step = np.cumsum(np.real(impulse))

        rho_step = np.clip(rho_step, -0.99, 0.99)
        denom = 1.0 - rho_step
        z_t = np.where(np.abs(denom) < 1e-6, np.nan, zref_ohm * (1.0 + rho_step) / denom)

        n_plot = n_time // 2
        time_s = np.arange(n_plot, dtype=float) * dt
        return time_s, z_t[:n_plot]

    def _build_frequency_window(self, size: int) -> np.ndarray:
        if size <= 0:
            return np.array([], dtype=float)

        window_name = self._window_combo.currentText().strip().lower()
        if window_name == "none":
            return np.ones(size, dtype=float)

        # Preserve DC/low-frequency bins and taper only the high-frequency edge.
        w = np.ones(size, dtype=float)
        if size < 8:
            return w

        taper = max(4, int(round(size * 0.15)))
        taper = min(taper, size - 1)
        two_sided_len = max(2 * taper, 8)

        if window_name == "hamming":
            tail = signal.windows.hamming(two_sided_len, sym=False)[taper:]
        elif window_name == "blackman":
            tail = signal.windows.blackman(two_sided_len, sym=False)[taper:]
        elif window_name == "flattop":
            tail = signal.windows.flattop(two_sided_len, sym=False)[taper:]
        else:
            tail = signal.windows.hann(two_sided_len, sym=False)[taper:]

        tail = np.asarray(tail, dtype=float)
        if tail.size != taper:
            tail = np.resize(tail, taper)
        if tail[0] != 0:
            tail = tail / tail[0]
        tail = np.clip(tail, 0.0, 1.0)
        w[-taper:] = tail
        w[0] = 1.0
        return w