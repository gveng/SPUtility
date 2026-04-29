from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from sparams_utility.circuit_solver import TransientSimResult

_AXIS_PEN = pg.mkPen(color="#222222", width=1)
_TRACE_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#9467bd",
    "#e377c2",
]


class TransientResultWindow(QMainWindow):
    def __init__(self, result: TransientSimResult, *, title: str = "Transient Result", parent=None) -> None:
        super().__init__(parent)
        self._result = result
        self.setWindowTitle(title)
        self.resize(1180, 760)
        app = QApplication.instance()
        if app is not None:
            self.setWindowIcon(app.windowIcon())

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._plot_widget.setLabel("bottom", "Time", units="ns")
        self._plot_widget.setLabel("left", "Voltage", units="V")
        plot_item = self._plot_widget.getPlotItem()
        for side in ("bottom", "left", "top", "right"):
            axis = plot_item.getAxis(side)
            axis.setPen(_AXIS_PEN)
            axis.setTextPen(_AXIS_PEN)
        plot_item.getAxis("top").setStyle(showValues=False)
        plot_item.getAxis("right").setStyle(showValues=False)
        plot_item.showAxes(True, showValues=(True, False, False, True))
        self._legend = self._plot_widget.addLegend(offset=(10, 10))

        warning_text = "Transient simulation completed."
        if result.warnings:
            warning_text = "Warnings: " + " | ".join(result.warnings)
        self._warning_label = QLabel(warning_text)
        self._warning_label.setWordWrap(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._warning_label)
        layout.addWidget(self._plot_widget)
        self.setCentralWidget(container)

        time_ns = result.time_s * 1e9
        for index, trace in enumerate(result.traces):
            color = _TRACE_COLORS[index % len(_TRACE_COLORS)]
            self._plot_widget.plot(
                time_ns,
                trace.waveform_v,
                pen=pg.mkPen(color=color, width=2),
                name=trace.label,
            )

    def update_result(self, result: TransientSimResult) -> None:
        """Replace existing curves with the ones coming from a new run."""
        self._result = result
        plot_item = self._plot_widget.getPlotItem()
        plot_item.clear()
        if self._legend is not None:
            try:
                self._legend.scene().removeItem(self._legend)
            except Exception:
                pass
        self._legend = self._plot_widget.addLegend(offset=(10, 10))

        warning_text = "Transient simulation completed."
        if result.warnings:
            warning_text = "Warnings: " + " | ".join(result.warnings)
        self._warning_label.setText(warning_text)

        time_ns = result.time_s * 1e9
        for index, trace in enumerate(result.traces):
            color = _TRACE_COLORS[index % len(_TRACE_COLORS)]
            self._plot_widget.plot(
                time_ns,
                trace.waveform_v,
                pen=pg.mkPen(color=color, width=2),
                name=trace.label,
            )
